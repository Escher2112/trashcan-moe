# Big mixture-of-experts models on the 2013 Mac Pro

A recipe for [TrashCanAI](https://github.com/devwithzachary/TrashCanAI), measured on a stand-in machine. I do not own a trash can,
so every number for your machine below is an estimate. I would love to see the real ones.

## Short version

Your Mac Pro talks at 13.0 tok/s on the 7B (both D500s), 6.4 on the 14B and 2.5 on the 32B. Those are all dense models: a dense model
reads its whole self for every token, so once it spills out of your 6 GB of card memory it slows down in step with its size.

A mixture-of-experts (MoE) model is big on disk but only runs a small slice of itself per token. That suits a machine with a lot of
slower memory and small graphics cards, which is exactly what the trash can is.

| Model | Total size | Runs on | Talking speed, estimated for your machine (8 threads) |
|---|---|---|---|
| Gemma 4 26B-A4B (about 4B active per token) | 26B, 14.2 GB file | CPU and RAM only | about 12.5 to 13.3 tok/s |
| Qwen 3.6 35B-A3B (about 3B active per token) | 35B, 22.4 GB file | CPU and RAM only | about 11.3 to 12.0 tok/s |

So: roughly your 7B's write speed from a model five times the size, close to five times the speed of your 32B, and both D500s stay
free. It needs no rebuild. Your pinned llama.cpp commit (`0f35d6d7b`) already runs both models; I tested on that exact commit.

## The recipe

### 1. Download the two files

```bash
cd ~/models
wget -c https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF/resolve/main/google_gemma-4-26B-A4B-it-IQ4_XS.gguf
wget -c https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

14.2 GB and 22.4 GB. Both models are Apache 2.0.

### 2. Point two presets at them

In `~/.trashcan-models.ini`, replace the `[trash-deep]` and `[trash-heavy]` sections with these. Your `[*]` block (12 threads, 4096
context, jinja) stays as it is.

```ini
[trash-deep]
; Gemma 4 26B-A4B: 26B total, about 4B active per token. Lives in RAM, runs on the Xeon.
model = /home/YOUR_USER/models/google_gemma-4-26B-A4B-it-IQ4_XS.gguf
n-gpu-layers = 0
reasoning = off

[trash-heavy]
; Qwen 3.6 35B-A3B: 35B total, about 3B active per token. Lives in RAM, runs on the Xeon.
model = /home/YOUR_USER/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
n-gpu-layers = 0
reasoning = off
```

Because the preset names are unchanged, `trash-ai code --deep ...` and `trash-ai code --heavy ...` use the new models with no code
change. If you would rather keep the 14B and 32B, give these two sections new names and call them by name through the API.

`reasoning = off` matters. Both are thinking models; left on, a short job like spellcheck spends its tokens on hidden reasoning before
it answers. For a hard coding question, a second preset with `reasoning = on` is worth trying.

### 3. Run it and measure it

```bash
~/serve.sh
trash-ai code --heavy "Write a Kotlin Jetpack Compose function that shows CPU temperature in a Material 3 Card"

# your own benchmark protocol, CPU only, 8 threads (to compare with your Table 2) and 12 threads
~/llama.cpp/build/bin/llama-bench -m ~/models/google_gemma-4-26B-A4B-it-IQ4_XS.gguf -n 128 -p 512 -ngl 0 -t 8,12 -r 3 -o md
~/llama.cpp/build/bin/llama-bench -m ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf      -n 128 -p 512 -ngl 0 -t 8,12 -r 3 -o md
```

The first request to each preset includes loading the file, which took 30 to 50 seconds from a SATA SSD on my machine.

## How I got the numbers

**The stand-in:** a Dell Precision 5810, Xeon E5-2687W v3 (10 Haswell cores, hyperthreading off), 32 GB DDR4-2133 in four channels,
Linux, transparent huge pages `always`.

**Making it behave like Ivy Bridge:** llama.cpp at your commit `0f35d6d7b`, built with your flags plus three more so that the newer
chip cannot use anything yours lacks:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_AVX=ON -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=ON -DGGML_BMI2=OFF \
  -DCMAKE_C_FLAGS=-march=ivybridge -DCMAKE_CXX_FLAGS=-march=ivybridge
```

`objdump` on the built libraries finds no FMA and no AVX2 integer instructions. No graphics card was used for anything on this page.

**Calibration: your models, your benchmark, your CPU table.** `llama-bench -n 128 -p 512 -ngl 0 -t 8 -r 3`:

| Model | Your Mac Pro, talks | Stand-in, talks | Yours ÷ mine |
|---|---|---|---|
| Qwen 2.5 Coder 1.5B Q8_0 | 27.1 | 26.71 | 1.01 |
| Qwen 2.5 Coder 7B Q4_K_M | 8.94 | 9.45 | 0.95 |

Talking speed on the stand-in lands within about 5% of your published numbers. That band (0.95 to 1.01) is what turns my MoE numbers
into the estimates at the top.

**The MoE models on the stand-in, CPU only:**

| Model | Threads | Reads (pp512) | Talks (tg128) |
|---|---|---|---|
| Gemma 4 26B-A4B IQ4_XS | 8 | 33.40 | 13.15 |
| Gemma 4 26B-A4B IQ4_XS | 10 | 38.22 | 14.27 |
| Gemma 4 26B-A4B Q4_K_M | 8 | 31.47 | 12.38 |
| Gemma 4 26B-A4B Q4_K_M | 10 | 36.18 | 13.44 |
| Qwen 3.6 35B-A3B UD-Q4_K_XL | 8 | 38.71 | 11.86 |
| Qwen 3.6 35B-A3B UD-Q4_K_XL | 10 | 43.88 | 12.43 |

**Your own task prompts** (from `scripts/benchmark.py`), through `llama-server -ngl 0 -t 8 -c 4096 --jinja --reasoning off`:

| Model | Task | Total | Wrote |
|---|---|---|---|
| Gemma 4 26B | grammar_spellcheck | 6.3 s | 43 tokens at 12.83 tok/s |
| Gemma 4 26B | coding_android | 34.5 s | 400 tokens at 12.30 tok/s |
| Qwen 3.6 35B | grammar_spellcheck | 6.1 s | 43 tokens at 11.72 tok/s |
| Qwen 3.6 35B | coding_android | 35.9 s | 400 tokens at 11.67 tok/s |

Both returned the same correct spellcheck and a sensible `TrashCanDashboard` composable (stopped at my 400-token cap). I also ran the
two preset sections above through `llama-server --models-preset ... --models-max 1` and swapped between them by name; they load,
answer and swap cleanly.

## What I could not test

1. **It is not a Mac Pro.** Different chip generation, DDR4 instead of DDR3, 10 cores instead of 12. The calibration says talking
   speed transfers well. It says nothing certain about anything else.
2. **Reading (prompt) speed did not calibrate, and I think I know why.** Your Table 2 "pure CPU" prompt numbers are 3 to 4 times
   mine (59.5 against 20.85 on the 7B) while the talking numbers match. A Vulkan build still hands large-batch work to the cards at
   `-ngl 0` unless told not to (`--no-op-offload`). Your own table hints at it: the 32B reads at 9.05 tok/s on 8 threads and 9.06 on
   12, and a CPU-bound job would scale with threads. If that is right, your cards will also help these two models read, or get in
   the way; I cannot tell which. One command settles it:
   `llama-bench -m <file> -n 128 -p 512 -ngl 0 -t 12 -nopo 0,1 -r 3 -o md`. In a preset the key is `no-op-offload = true`.
3. **Putting part of the model on the cards.** llama.cpp can keep the experts in RAM and put everything else on the GPUs:
   `n-gpu-layers = 99` plus `cpu-moe = true` (and your usual `split-mode = layer`, `tensor-split = 1,1`, `flash-attn = on`). On an
   old 8 GB NVIDIA card with CUDA, a split like that took Gemma from about 15 to about 21 tok/s for me. I have not run it on Vulkan
   with two D500s, and the PCIe bounce you measured on your 32B hybrid may eat the gain. Worth one benchmark:
   add `-ngl 99 -ncmoe 99 -sm layer -ts 1/1 -fa 1` to the llama-bench line.
4. **12 threads.** I only have 10 cores. Going from 8 to 10 threads gave me 5 to 9% more; you measured +22% from 8 to 12 on your 32B.
5. **Vision.** Gemma 4 can see (the `mmproj-google_gemma-4-26B-A4B-it-f16.gguf` file, 1.19 GB, same download page). I did not time
   image encoding on the CPU, so I do not know how it compares with Moondream2 on your cards.
6. **Long prompts.** On the stand-in's CPU these two read at 30 to 40 tok/s, so a 2,000-token file takes about a minute before the
   first word. If your machine reads them at a similar rate, that is still quicker than your 14B (27.6) and 32B (9.06) read today,
   but see point 2: reading is the number I trust least.

## Why these two models

- **Qwen 3.6 35B-A3B for code.** Qwen's model card lists 73.4 on SWE-bench Verified (their own agent scaffold) and 51.5 on
  Terminal-Bench 2.0. In the same table Gemma 4 26B-A4B gets 17.4 and 34.2. Those are the vendor's numbers, but the gap is wide.
- **Gemma 4 26B-A4B for everything else.** Google's card lists 77.1% on LiveCodeBench v6 and 82.6% on MMLU Pro, it is a little
  faster than Qwen here, and it has a vision encoder.
- I did not run a quality head-to-head against Qwen 2.5 Coder. The claim on this page is about speed for size; judge the answers
  on your own Kotlin.

Model cards: [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) ·
[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)

## If you try it

Post your `llama-bench` tables, whatever they say. If the estimates are off I want to know by how much, and if the card options in
"What I could not test" help or hurt on real D500s, that is the part nobody has measured.
