import os
import json
import time
import re
from transformers import TrainerCallback
from llava.utils import rank0_print
from llava.constants import IGNORE_INDEX

# Additional imports for checkpoint-triggered evaluation
import re
import json
import time
import os
import sys
import subprocess
import threading
import pathlib


class BabyLMCheckpointCallback(TrainerCallback):
    """
    Callback per BabyLM Challenge - SOLUZIONE MINIMALE

    Conta le parole accedendo direttamente al dataset originale invece di
    passarle attraverso il training loop. Richiede solo il dataset come parametro.

    Salva checkpoint con frequenza variabile:
        * Ogni 1M parole fino a 10M
        * Ogni 10M parole fino a 100M
        * Ogni 100M parole fino a 1B
    """

    def __init__(self, tokenizer, dataset, target_word_limit=None):
        """
        Args:
            tokenizer: Tokenizer del modello
            dataset: Il dataset di training (LazySupervisedDataset)
            target_word_limit: Limite massimo di parole (default: 1B)
        """
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.target_word_limit = target_word_limit if target_word_limit else 1_000_000_000
        self.total_words_seen = 0
        self.saved_milestones = set()
        self.trainer = None

        # Regex per pulire token speciali
        self.token_remover = re.compile(r'<[^>]+>')

        # Cache per evitare di ricontare gli stessi sample
        self.processed_samples = set()

        # Genera milestones
        self.milestones = self._generate_milestones()

        rank0_print(f"🎯 BabyLM Callback inizializzato (accesso diretto al dataset)")
        rank0_print(f"   Dataset samples: {len(dataset)}")
        rank0_print(f"   Milestones: {len(self.milestones)} checkpoint")
        rank0_print(f"   Limite: {self.target_word_limit:,} parole")

    def _generate_milestones(self):
        """Genera lista di milestones per i checkpoint."""
        milestones = []

        # Ogni 1M fino a 10M
        milestones.extend(range(1_000_000, 10_000_001, 1_000_000))

        # Ogni 10M fino a 100M
        if self.target_word_limit > 10_000_000:
            milestones.extend(range(20_000_000, 100_000_001, 10_000_000))

        # Ogni 100M fino a limite
        if self.target_word_limit > 100_000_000:
            milestones.extend(range(200_000_000, self.target_word_limit + 1, 100_000_000))

        # Filtra solo milestones <= limite e rimuovi duplicati
        milestones = sorted(list(set([m for m in milestones if m <= self.target_word_limit])))

        return milestones

    def _count_words_from_text(self, text):
        """
        Conta parole dal testo originale (stesso metodo dello script di analisi).

        Args:
            text (str): Testo da contare

        Returns:
            int: numero di parole
        """
        # Rimuovi token speciali
        clean_text = self.token_remover.sub('', text)
        clean_text = clean_text.replace('\\n', ' ')

        # Conta parole (stesso algoritmo dello script esterno)
        words = ''.join(
            c.lower() if c.isalnum() else ' '
            for c in clean_text
        ).split()

        return len(words)

    def _count_words_in_sample(self, sample_dict):
        """
        Conta parole in un sample del dataset (solo risposte GPT/assistant).

        Args:
            sample_dict: dict con campo 'conversations'

        Returns:
            int: numero di parole
        """
        word_count = 0
        conversations = sample_dict.get('conversations', [])

        for turn in conversations:
            # Conta solo messaggi da "gpt" o "assistant"
            if turn.get('from') in ['gpt', 'assistant']:
                text = turn.get('value', '')
                word_count += self._count_words_from_text(text)

        return word_count

    def _estimate_samples_per_batch(self, args=None):
        """
        Stima il batch size effettivo.

        Args:
            args: TrainingArguments (opzionale, se None usa self.trainer.args)

        Returns:
            int: batch size effettivo
        """
        # Prova prima con args passato direttamente
        if args is not None:
            per_device_bs = getattr(args, 'per_device_train_batch_size', 1)
            gradient_accum = getattr(args, 'gradient_accumulation_steps', 1)
            effective_bs = per_device_bs * gradient_accum
            if effective_bs > 1:
                return effective_bs

        # Fallback: prova con self.trainer
        if self.trainer is not None and hasattr(self.trainer, 'args'):
            args = self.trainer.args
            per_device_bs = getattr(args, 'per_device_train_batch_size', 1)
            gradient_accum = getattr(args, 'gradient_accumulation_steps', 1)
            effective_bs = per_device_bs * gradient_accum
            if effective_bs > 1:
                return effective_bs

        return 1

    def _get_current_sample_range(self, step):
        """
        Calcola quali sample del dataset sono stati processati fino a questo step.

        Assume che il training proceda sequenzialmente attraverso il dataset.

        Args:
            step: step di training corrente

        Returns:
            tuple: (start_idx, end_idx) degli indici dei sample
        """
        batch_size = self._estimate_samples_per_batch()
        start_idx = step * batch_size
        end_idx = min((step + 1) * batch_size, len(self.dataset))

        return start_idx, end_idx

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """All'inizio del training, salva il riferimento al trainer."""
        self.trainer = kwargs.get('trainer', None)

        # Aspetta che il trainer sia pronto prima di stimare il batch size
        if self.trainer is not None and hasattr(self.trainer, 'args'):
            batch_size = self._estimate_samples_per_batch()
            rank0_print(f"📦 Batch size effettivo: {batch_size}")
            rank0_print(f"📊 Samples totali: {len(self.dataset)}")

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """
        Dopo ogni step, conta le parole dai sample del dataset processati in questo step.
        """
        if self.trainer is None or self.dataset is None:
            return control

        # Calcola quali sample sono stati processati in questo step
        start_idx, end_idx = self._get_current_sample_range(state.global_step - 1)

        # DEBUG: Stampa info sui primi 3 steps
        if state.global_step <= 3:
            rank0_print(f"\n🔍 DEBUG Step {state.global_step}:")
            rank0_print(f"   Sample range: {start_idx} → {end_idx}")
            rank0_print(f"   Batch size: {self._estimate_samples_per_batch()}")

        # Conta parole solo nei sample non ancora processati
        words_in_step = 0
        samples_counted = 0

        for idx in range(start_idx, end_idx):
            # Skip se già processato (per resume o multi-epoch)
            if idx in self.processed_samples:
                continue

            if idx < len(self.dataset.list_data_dict):
                sample = self.dataset.list_data_dict[idx]
                words = self._count_words_in_sample(sample)
                words_in_step += words
                self.processed_samples.add(idx)
                samples_counted += 1

                # DEBUG: Mostra primo sample
                if state.global_step <= 3 and samples_counted == 1:
                    rank0_print(f"   Primo sample (idx={idx}): {words} parole")

        self.total_words_seen += words_in_step

        # DEBUG: Info dettagliato primi 3 steps
        if state.global_step <= 3:
            rank0_print(f"   Parole in questo step: {words_in_step:,}")
            rank0_print(f"   Samples contati: {samples_counted}")
            rank0_print(f"   Totale accumulato: {self.total_words_seen:,}\n")

        # Log ogni 100 steps
        if state.global_step % 100 == 0:
            rank0_print(
                f"📊 Step {state.global_step}: {self.total_words_seen:,} parole "
                f"(+{words_in_step:,} da {samples_counted} samples)"
            )

        # Check milestones
        for milestone in self.milestones:
            if self.total_words_seen >= milestone and milestone not in self.saved_milestones:
                control.should_save = True
                self.saved_milestones.add(milestone)

                rank0_print(f"\n{'='*60}")
                rank0_print(f"🎯 MILESTONE: {self._format_count(milestone)}")
                rank0_print(f"   Totale: {self.total_words_seen:,} parole")
                rank0_print(f"   Samples processati: {len(self.processed_samples)}")
                rank0_print(f"{'='*60}\n")

                if args.process_index == 0 or args.local_rank in [-1, 0]:
                    state.babylm_pending_metadata = {
                        'milestone': milestone,
                        'total_words': self.total_words_seen,
                        'step': state.global_step,
                        'samples_processed': len(self.processed_samples)
                    }
                break

        # Check limite
        if self.total_words_seen >= self.target_word_limit:
            rank0_print(f"\n🛑 Limite raggiunto: {self.target_word_limit:,} parole")
            rank0_print(f"   Samples processati: {len(self.processed_samples)}/{len(self.dataset)}")
            control.should_training_stop = True

        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Conta le parole quando viene fatto il log (succede ogni N steps).
        """
        if logs is None:
            return control

        # Conta le parole dal dataset
        if self.dataset is not None:
            # Calcola quali sample sono stati processati fino a questo step
            batch_size = self._estimate_samples_per_batch(args)
            start_idx = 0
            end_idx = min(state.global_step * batch_size, len(self.dataset.list_data_dict))

            # Conta parole solo nei sample non ancora processati
            words_in_batch = 0
            samples_counted = 0

            for idx in range(start_idx, end_idx):
                if idx in self.processed_samples:
                    continue

                if idx < len(self.dataset.list_data_dict):
                    sample = self.dataset.list_data_dict[idx]
                    words = self._count_words_in_sample(sample)
                    words_in_batch += words
                    self.processed_samples.add(idx)
                    samples_counted += 1

            if words_in_batch > 0:
                self.total_words_seen += words_in_batch

            # Log ogni 1000 steps
            if state.global_step % 1000 == 0:
                rank0_print(
                    f"📊 Step {state.global_step}: {self.total_words_seen:,} parole "
                    f"(samples processati: {len(self.processed_samples):,})"
                )

        # Aggiungi al log
        logs['babylm/words_seen'] = self.total_words_seen
        logs['babylm/milestones_saved'] = len(self.saved_milestones)
        logs['babylm/samples_processed'] = len(self.processed_samples)

        # Check milestones
        for milestone in self.milestones:
            if self.total_words_seen >= milestone and milestone not in self.saved_milestones:
                control.should_save = True
                self.saved_milestones.add(milestone)

                rank0_print(f"\n{'='*60}")
                rank0_print(f"🎯 MILESTONE: {self._format_count(milestone)}")
                rank0_print(f"   Totale: {self.total_words_seen:,} parole")
                rank0_print(f"   Samples processati: {len(self.processed_samples):,}")
                rank0_print(f"{'='*60}\n")

                if args.process_index == 0 or args.local_rank in [-1, 0]:
                    state.babylm_pending_metadata = {
                        'milestone': milestone,
                        'total_words': self.total_words_seen,
                        'step': state.global_step,
                        'samples_processed': len(self.processed_samples)
                    }
                break

        # Check limite
        if self.total_words_seen >= self.target_word_limit:
            rank0_print(f"\n🛑 Limite raggiunto: {self.target_word_limit:,} parole")
            control.should_training_stop = True

        return control

    def on_save(self, args, state, control, **kwargs):
        """Rinomina checkpoint con milestone."""
        if not (args.process_index == 0 or args.local_rank in [-1, 0]):
            return control

        if not hasattr(state, 'babylm_pending_metadata'):
            return control

        meta = state.babylm_pending_metadata
        original_dir = os.path.join(args.output_dir, f"checkpoint-{meta['step']}")

        # Nuovo nome con milestone
        milestone_suffix = self._format_count(meta['milestone'])
        new_name = f"checkpoint-{meta['step']}-{milestone_suffix}words"
        new_dir = os.path.join(args.output_dir, new_name)

        # Attendi creazione checkpoint
        for _ in range(30):
            if os.path.exists(original_dir):
                break
            time.sleep(1)

        # Rinomina
        if os.path.exists(original_dir):
            try:
                rank0_print(f"🔄 Rinominando: {new_name}")
                os.rename(original_dir, new_dir)
                self._save_metadata(new_dir, meta)
            except Exception as e:
                rank0_print(f"❌ Errore rinomina: {e}")
                self._save_metadata(original_dir, meta)

                # If we couldn't rename, fall back to original_dir for eval
                new_dir = original_dir

        # --- Start background BLIMP fast evaluation on this checkpoint (non-blocking) ---
        try:
            # Only start eval from process 0 to avoid duplicates
            if args.process_index == 0 or getattr(args, 'local_rank', -1) in [-1, 0]:
                checkpoint_path = os.path.abspath(new_dir)
                checkpoint_name = os.path.basename(checkpoint_path)
                backend = os.environ.get('BABYLM_EVAL_BACKEND', 'causal')
                # Use trainer args.output_dir if available for logging
                train_output_dir = getattr(args, 'output_dir', None) or getattr(self.trainer.args, 'output_dir', None) if self.trainer is not None else None

                def _eval_and_log():
                    rank0_print(f"▶️ Launching BLIMP fast eval for checkpoint: {checkpoint_name}")
                    log_path = os.path.join(checkpoint_path, 'blimp_eval_output.log')
                    eval_output_dir = os.path.join(checkpoint_path, 'eval_results')
                    os.makedirs(eval_output_dir, exist_ok=True)

                    # Build evaluation command (BLIMP fast only)
                    EVAL_FAST_BLIMP = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'babylm_eval', 'evaluation_data', 'fast_eval', 'blimp_fast')
                    cmd = [
                        sys.executable, '-u', '-m', 'evaluation_pipeline.sentence_zero_shot.run',
                        '--model_path_or_name', checkpoint_path,
                        '--backend', backend,
                        '--task', 'blimp',
                        '--data_path', EVAL_FAST_BLIMP,
                        '--save_predictions',
                        '--revision_name', checkpoint_name,
                        '--output_dir', eval_output_dir
                    ]

                    with open(log_path, 'w') as outf:
                        # Ensure the evaluation package is importable by setting PYTHONPATH
                        env_copy = os.environ.copy()
                        eval_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'babylm_eval'))
                        existing = env_copy.get('PYTHONPATH', '')
                        env_copy['PYTHONPATH'] = eval_root + (':' + existing if existing else '')
                        proc = subprocess.Popen(cmd, stdout=outf, stderr=subprocess.STDOUT, env=env_copy)
                        proc.wait()

                    # Attempt to read the created report
                    report_path = os.path.join(eval_output_dir, os.path.basename(checkpoint_path), checkpoint_name, 'zero_shot', backend, 'blimp', 'blimp_fast', 'best_temperature_report.txt')
                    accuracy = None
                    if os.path.exists(report_path):
                        try:
                            with open(report_path, 'r') as rf:
                                lines = rf.read().splitlines()
                            for i, line in enumerate(lines):
                                if line.strip().upper().startswith('### AVERAGE') and i+1 < len(lines):
                                    try:
                                        accuracy = float(lines[i+1].strip())
                                    except Exception:
                                        pass
                                    break
                        except Exception as e:
                            rank0_print(f"❌ Error reading BLIMP report: {e}")

                    # Fallback: try to parse the log for a numeric line
                    if accuracy is None:
                        try:
                            with open(log_path, 'r') as lf:
                                for ln in reversed(lf.readlines()[-200:]):
                                    stripped = ln.strip()
                                    try:
                                        # lines like: '1.0\t52.30' may exist (temp\tacc)
                                        parts = stripped.split()[-1]
                                        val = float(parts.strip())
                                        accuracy = val
                                        break
                                    except Exception:
                                        continue
                        except Exception:
                            pass

                    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    summary_line = f"{timestamp} | BLIMP_FAST | checkpoint={checkpoint_name} | accuracy={accuracy if accuracy is not None else 'N/A'} | report={report_path} | log={log_path}"

                    # Append to a training-visible log (in trainer output dir if available)
                    if train_output_dir is not None:
                        try:
                            out_log = os.path.join(train_output_dir, 'babylm_blimp_eval.log')
                            with open(out_log, 'a') as f:
                                f.write(summary_line + '\n')
                        except Exception as e:
                            rank0_print(f"❌ Could not write to train eval log: {e}")

                    # Always print to trainer stdout so it appears in training logs
                    rank0_print(f"📈 {summary_line}")

                # Start thread (non-blocking)
                t = threading.Thread(target=_eval_and_log, daemon=True)
                t.start()
        except Exception as e:
            rank0_print(f"❌ Failed to launch BLIMP eval thread: {e}")

        # Clean up pending metadata flag and return
        delattr(state, 'babylm_pending_metadata')
        return control

    def _format_count(self, count):
        """Formatta numero (es. 1.2M, 500k)."""
        if count >= 1_000_000:
            val = count / 1_000_000
            return f"{int(val)}M" if val.is_integer() else f"{val:.1f}M"
        elif count >= 1_000:
            val = count / 1_000
            return f"{int(val)}k" if val.is_integer() else f"{val:.1f}k"
        return str(count)

    def _save_metadata(self, checkpoint_dir, meta):
        """Salva metadata JSON."""
        metadata = {
            "babylm_compliant": True,
            "milestone": meta['milestone'],
            "total_words_seen": meta['total_words'],
            "samples_processed": meta.get('samples_processed', 0),
            "global_step": meta['step'],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            path = os.path.join(checkpoint_dir, "babylm_metadata.json")
            with open(path, "w") as f:
                json.dump(metadata, f, indent=2)
            rank0_print(f"✅ Metadata salvati: {path}")
        except Exception as e:
            rank0_print(f"❌ Errore salvataggio metadata: {e}")

    def state_dict(self):
        """Stato per resume."""
        return {
            "total_words_seen": self.total_words_seen,
            "saved_milestones": list(self.saved_milestones),
            "processed_samples": list(self.processed_samples),
        }

    def load_state_dict(self, state_dict):
        """Ripristina stato."""
        self.total_words_seen = state_dict.get("total_words_seen", 0)
        self.saved_milestones = set(state_dict.get("saved_milestones", []))
        self.processed_samples = set(state_dict.get("processed_samples", []))

        rank0_print(f"📂 Ripristinato stato callback:")
        rank0_print(f"   Parole: {self.total_words_seen:,}")
        rank0_print(f"   Milestones: {len(self.saved_milestones)}")
        rank0_print(f"   Samples: {len(self.processed_samples)}")
