using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO.MemoryMappedFiles;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Text;
#pragma warning disable CA2014

namespace llama2.cs;

[SuppressMessage("ReSharper", "StackAllocInsideLoop")]
public static class Program
{
    private static long _rngSeed;

    public static void Main(string[] args)
    {
        int argc = args.Length;
        string? checkpoint = null;
        float temperature = 1.0f; // 0.0 = greedy deterministic. 1.0 = original. don't set higher
        float topp = 0.9f; // top-p in nucleus sampling
        SetSeed((uint) DateTime.UtcNow.Ticks);
        int steps = 256; // number of steps to run for
        string? prompt = null; // prompt string

        if (argc >= 1)
            checkpoint = args[0];
        else
        {
            ErrorUsage();
            return;
        }

        void ErrorUsage()
        {
            Console.WriteLine("Usage:   run <checkpoint> [options]");
            Console.WriteLine("Example: run model.bin -n 256 -i \"Once upon a time\"");
            Console.WriteLine("Options:");
            Console.WriteLine("  -t <float>  temperature, default 1.0");
            Console.WriteLine("  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off");
            Console.WriteLine("  -s <int>    random seed, default time(NULL)");
            Console.WriteLine("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
            Console.WriteLine("  -i <string> input prompt");
        }

        for (int i = 1; i < argc; i += 2)
        {
            if (args[i][1] == 't')
                temperature = float.Parse(args[i + 1]);
            else if (args[i][1] == 'p')
                topp = float.Parse(args[i + 1]);
            else if (args[i][1] == 's')
                _rngSeed = int.Parse(args[i + 1]);
            else if (args[i][1] == 'n')
                steps = int.Parse(args[i + 1]);
            else if (args[i][1] == 'i')
            {
                prompt = args[i + 1];
            }
            else
                ErrorUsage();
        }

        if (_rngSeed == 0)
        {
            Console.WriteLine("Cannot use seed=0 because of the rng alg used\n");
            return;
        }

        // read in the model.bin file
        Config config;
        TransformerWeights weights;
        {
            try
            {
                using FileStream fileStream = new FileStream(checkpoint, FileMode.Open, FileAccess.Read);
                // Read in the config header
                byte[] configBytes = new byte[Marshal.SizeOf(typeof(Config))];
                if (fileStream.Read(configBytes, 0, configBytes.Length) != configBytes.Length) Environment.Exit(1);

                GCHandle handle = GCHandle.Alloc(configBytes, GCHandleType.Pinned);
                try
                {
                    IntPtr pointer = handle.AddrOfPinnedObject();
                    config = (Config) Marshal.PtrToStructure(pointer, typeof(Config))!;
                }
                finally
                {
                    handle.Free();
                }

                // Negative vocab size is a hacky way of signaling unshared weights. Bit yikes.
                bool sharedWeights = config.vocab_size > 0;
                config.vocab_size = Math.Abs(config.vocab_size);

                // Figure out the file size
                var fileSize = fileStream.Length; // size of the checkpoint file in bytes

                using var memoryMappedFile = MemoryMappedFile.CreateFromFile(fileStream, null, fileSize,
                    MemoryMappedFileAccess.Read, HandleInheritability.None, false);
                long configSizeInBytes = Marshal.SizeOf(typeof(Config));
                using var accessor = memoryMappedFile.CreateViewAccessor(configSizeInBytes,
                    fileSize - configSizeInBytes, MemoryMappedFileAccess.Read);
                weights = new TransformerWeights();

                CheckpointInitWeights(ref weights, ref config, accessor, sharedWeights);
            }
            catch (FileNotFoundException)
            {
                Console.Error.WriteLine($"Couldn't open file {checkpoint}");
                return;
            }
            catch (Exception e)
            {
                Console.Error.WriteLine($"Couldn't read {checkpoint}: {e.Message}");
                return;
            }
        }

        // right now we cannot run for more than config.seq_len steps
        if (steps <= 0 || steps > config.seq_len) steps = config.seq_len;

        // read in the tokenizer.bin file
        string[] vocab = new string[config.vocab_size];
        float[] vocabScores = new float[config.vocab_size];
        int maxTokenLength;

        using (FileStream fs = new FileStream(@"tokenizer.bin", FileMode.Open,
                   FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            try
            {
                maxTokenLength = reader.ReadInt32();

                for (int i = 0; i < config.vocab_size; i++)
                {
                    vocabScores[i] = reader.ReadSingle();

                    int len = reader.ReadInt32();
                    Span<byte> buffer = stackalloc byte[len]; // stack allocate buffer, assumes len is small
                    _ = reader.Read(buffer);

                    vocab[i] = Encoding.UTF8.GetString(buffer);
                }
            }
            catch (EndOfStreamException)
            {
                Console.Error.WriteLine("failed read");
                return;
            }
        }

        // create and init the application RunState
        RunState state = InitializeRunState(config);

        // process the prompt, if any
        int[]? promptTokens = null;
        int numPromptTokens = 0;
        if (!string.IsNullOrEmpty(prompt))
        {
            promptTokens = new int[prompt.Length];
            BpeEncode(prompt, vocab, vocabScores, config.vocab_size, maxTokenLength, ref promptTokens,
                ref numPromptTokens);
        }

        // start the main loop
        int token = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0; // position in the sequence
        Stopwatch timer = new Stopwatch();
        timer.Start();

        while (pos < steps)
        {
            // forward the transformer to get logits for the next token
            Transformer(token, pos, config, state, weights);

            // advance the state state machine
            int next; // will store the next token in the sequence
            if (pos < numPromptTokens)
            {
                // if we are still processing the input prompt, force the next prompt token
                next = promptTokens![pos];
            }
            else
            {
                // sample the next token
                if (temperature == 0.0f)
                {
                    // greedy argmax sampling: take the token with the highest probability
                    next = Argmax(state.logits, config.vocab_size);
                }
                else
                {
                    // apply the temperature to the logits
                    for (int q = 0; q < config.vocab_size; q++) state.logits[q] /= temperature;

                    // apply softmax to the logits to get the probabilities for next token
                    Softmax(state.logits.AsSpan(0, config.vocab_size));
                    
                    // we sample from this distribution to get the next token
                    if (topp <= 0)
                        // simply sample from the predicted probability distribution
                        next = Sample(state.logits, config.vocab_size);
                    else
                        // top-p (nucleus) sampling, clamping the least likely tokens to zero
                        next = SampleTopp(state.logits, config.vocab_size, topp, state.probindex);
                }
            }

            pos++;

            // data-dependent terminating condition: the BOS (1) token delimits sequences
            if (next == 1) break;

            // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
            string tokenStr = token == 1 && vocab[next][0] == ' ' ? vocab[next].TrimStart() : vocab[next];
            Console.Write(tokenStr);
            token = next;
        }

        timer.Stop();
        Console.WriteLine();

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1)
            Console.WriteLine(
                $"achieved tok/s: {(pos - 1) / timer.Elapsed.Seconds}, tokens : {pos - 1} time : {timer.Elapsed}");
    }

    private static void BpeEncode(string text, string[] vocab, float[] vocabScores, int vocabSize, int maxTokenLength,
        ref int[] tokens, ref int nTokens)
    {
        int StrLookup(string str, string[] vocab, int vocabSize)
        {
            for (int i = 0; i < vocabSize; i++)
                if (str == vocab[i])
                    return i;
            return -1;
        }

        StringBuilder strBuffer = new StringBuilder(maxTokenLength * 2 + 1); // *2 for concat, +1 for null terminator

        // first encode every individual byte in the input string
        nTokens = 0; // the number of tokens
        foreach (char c in text)
        {
            strBuffer.Clear();
            strBuffer.Append(c);

            int id = StrLookup(strBuffer.ToString(), vocab, vocabSize);
            if (id == -1)
            {
                Console.Error.WriteLine("not good");
                throw new Exception("Encoding error");
            }

            tokens[nTokens] = id;
            nTokens++;
        }

        // merge the best consecutive pair each iteration, according to the scores in vocab_scores
        while (true)
        {
            float bestScore = float.MinValue;
            int bestId = -1;
            int bestIdx = -1;

            for (int i = 0; i < nTokens - 1; i++)
            {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                strBuffer.Clear();
                strBuffer.Append(vocab[tokens[i]]);
                strBuffer.Append(vocab[tokens[i + 1]]);

                int id = StrLookup(strBuffer.ToString(), vocab, vocabSize);
                if (id != -1 && vocabScores[id] > bestScore)
                {
                    // this merge pair exists in vocab! record its score and position
                    bestScore = vocabScores[id];
                    bestId = id;
                    bestIdx = i;
                }
            }

            if (bestIdx == -1) break; // we couldn't find any more pairs to merge, so we're done

            // merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
            tokens[bestIdx] = bestId;
            
            // delete token at position bestIdx+1, shift the entire sequence back 1
            for (int i = bestIdx + 1; i < nTokens - 1; i++) tokens[i] = tokens[i + 1];

            nTokens--; // token length decreased
        }
    }


    // This method sets the seed for the RNG
    private static void SetSeed(long seed)
    {
        _rngSeed = seed;
    }

    private static int RandomU32()
    {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        _rngSeed ^= _rngSeed >> 12;
        _rngSeed ^= _rngSeed << 25;
        _rngSeed ^= _rngSeed >> 27;
        return (int) ((_rngSeed * 0x2545F4914F6CDD1DL) >> 32);
    }

    private static float RandomF32()
    {
        // random float32 in [0,1)
        return (RandomU32() >>> 8) / 16777216.0f;
    }

    private static int Argmax(float[] probabilities, int configVocabSize)
    {
        int maxI = 0;
        float maxP = probabilities[0];
        for (int i = 1; i < configVocabSize; i++)
            if (probabilities[i] > maxP)
            {
                maxI = i;
                maxP = probabilities[i];
            }

        return maxI;
    }


    private static int Sample(float[] probabilities, int configVocabSize)
    {
        float r = RandomF32();
        float cdf = 0.0f;
        for (int i = 0; i < configVocabSize; i++)
        {
            cdf += probabilities[i];
            if (r < cdf) return i;
        }

        return configVocabSize - 1;
    }

    private static int Compare(ProbIndex a, ProbIndex b)
    {
        if (a.Prob > b.Prob) return -1;
        if (a.Prob < b.Prob) return 1;
        return 0;
    }

    private static int SampleTopp(float[] probabilities, int configVocabSize, float topp, ProbIndex[] probindex)
    {
        for (int i = 0; i < configVocabSize; i++)
        {
            probindex[i].Index = i;
            probindex[i].Prob = probabilities[i];
        }

        Array.Sort(probindex, Compare);

        float cumulativeProb = 0.0f;
        int lastIdx = 0;
        for (int i = 0; i < configVocabSize; i++)
        {
            cumulativeProb += probindex[i].Prob;
            if (cumulativeProb > topp)
            {
                lastIdx = i;
                break;
            }
        }

        float r = RandomF32() * cumulativeProb;
        float cdf = 0.0f;
        for (int i = 0; i <= lastIdx; i++)
        {
            cdf += probindex[i].Prob;
            if (r < cdf) return probindex[i].Index;
        }

        return probindex[lastIdx].Index;
    }


    private static void Accum(float[] a, float[] b, int size)
    {
        for (int i = 0; i < size; i++) a[i] += b[i];
    }

    private static void Rmsnorm(float[] o, float[] x, Span<float> weight, int size)
    {
        // calculate sum of squares
        float ss = TensorPrimitives.SumOfSquares(x.AsSpan(0, size));
        
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / MathF.Sqrt(ss);

        // normalize and scale
        for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
    }

    private static void Softmax(Span<float> x)
    {
        // find max value (for numerical stability)
        float maxVal = TensorPrimitives.Max(x);
        
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = MathF.Exp(x[i] - maxVal);
            sum += x[i];
        }

        // normalize
        TensorPrimitives.Divide(x, sum, x);
    }

    private static void Matmul(float[] xout, float[] x, ArraySegment<float> w, int n, int d)
    {
        // W (d,n) @ x (n,) . xout (d,)
        Parallel.For(0, d, i =>
        {
            xout[i] = TensorPrimitives.Dot(w.AsSpan(i * n, n), x);
        });
    }


    private static void Transformer(int token, int pos, Config config, RunState state, TransformerWeights w)
    {
        // a few convenience variables
        int dim = config.dim;
        int hiddenDim = config.hidden_dim;
        int headSize = dim / config.n_heads;

        // copy the token embedding into x
        Array.Copy(w.token_embedding_table, token * dim, state.x, 0, dim);

        // forward all the layers
        for (int l = 0; l < config.n_layers; l++)
        {
            // attention rmsnorm
            Rmsnorm(state.xb, state.x, w.rms_att_weight[(l * dim)..], dim);

            // qkv matmuls for this position
            Matmul(state.q, state.xb, w.wq[(l * dim * dim)..], dim, dim);
            Matmul(state.k, state.xb, w.wk[(l * dim * dim)..], dim, dim);
            Matmul(state.v, state.xb, w.wv[(l * dim * dim)..], dim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            for (int i = 0; i < dim; i += 2)
            {
                float q0 = state.q[i];
                float q1 = state.q[i + 1];
                float k0 = state.k[i];
                float k1 = state.k[i + 1];
                float fcr = w.freq_cis_real[pos * headSize / 2 + i % headSize / 2];
                float fci = w.freq_cis_imag[pos * headSize / 2 + i % headSize / 2];
                state.q[i] = q0 * fcr - q1 * fci;
                state.q[i + 1] = q0 * fci + q1 * fcr;
                state.k[i] = k0 * fcr - k1 * fci;
                state.k[i + 1] = k0 * fci + k1 * fcr;
            }

            // save key,value at this time step (pos) to our kv cache
            int loff = l * config.seq_len * dim; // kv cache layer offset for convenience
            Array.Copy(state.k, 0, state.key_cache, loff + pos * dim, dim);
            Array.Copy(state.v, 0, state.value_cache, loff + pos * dim, dim);

            // multihead attention. iterate over all heads
            Parallel.For(0, config.n_heads, h =>
            {
                // get the query vector for this head
                int qOffset = h * headSize;

                // attention scores for this head
                Span<float> att = state.att.AsSpan(h * config.seq_len);

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++)
                {
                    // get the key vector for this head and at this timestep
                    int keyCacheOffset = loff + t * dim + h * headSize;

                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < headSize; i++) score += state.q[i + qOffset] * state.key_cache[i + keyCacheOffset];

                    score /= MathF.Sqrt(headSize);

                    // save the score to the attention buffer
                    att[t] = score;
                }

                // softmax the scores to get attention weights, from 0..pos inclusively
                Softmax(att[..(pos + 1)]);

                // weighted sum of the values, store back into xb
                int xbOffset = h * headSize;
                for (int i = xbOffset; i < xbOffset + headSize; i++) state.xb[i] = 0f;

                for (int t = 0; t <= pos; t++)
                {
                    // get the value vector for this head and at this timestep
                    int vOffset = loff + t * dim + h * headSize;

                    // get the attention weight for this timestep
                    float a = att[t];

                    // accumulate the weighted value into xb
                    for (int i = 0; i < headSize; i++)
                        state.xb[i + xbOffset] += a * state.value_cache[i + vOffset];
                }
            });

            // final matmul to get the output of the attention
            Matmul(state.xb2, state.xb, w.wo[(l * dim * dim)..], dim, dim);

            // residual connection back into x
            Accum(state.x, state.xb2, dim);

            // ffn rmsnorm
            Rmsnorm(state.xb, state.x, w.rms_ffn_weight[(l * dim)..], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            Matmul(state.hb, state.xb, w.w1[(l * dim * hiddenDim)..], dim, hiddenDim);
            Matmul(state.hb2, state.xb, w.w3[(l * dim * hiddenDim)..], dim, hiddenDim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (int i = 0; i < hiddenDim; i++)
                state.hb[i] *= 1.0f / (1.0f + MathF.Exp(-state.hb[i]));

            // elementwise multiply with w3(x)
            TensorPrimitives.Multiply(state.hb.AsSpan(0, hiddenDim), state.hb2.AsSpan(0, hiddenDim), state.hb.AsSpan(0, hiddenDim));

            // final matmul to get the output of the ffn
            Matmul(state.xb, state.hb, w.w2[(l * dim * hiddenDim)..], hiddenDim, dim);

            // residual connection
            Accum(state.x, state.xb, dim);
        }

        // final rmsnorm
        Rmsnorm(state.x, state.x, w.rms_final_weight, dim);

        // classifier into logits
        Matmul(state.logits, state.x, w.wcls, config.dim, config.vocab_size);
    }

    private static void CheckpointInitWeights(ref TransformerWeights w, ref Config p, MemoryMappedViewAccessor accessor,
        bool sharedWeights)
    {
        long offset = 0;

        w.token_embedding_table = ReadFloatArray(accessor, ref offset, p.vocab_size * p.dim);
        w.rms_att_weight = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim);
        w.wq = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.wk = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.wv = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.wo = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.rms_ffn_weight = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim);
        w.w1 = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.hidden_dim);
        w.w2 = ReadFloatArray(accessor, ref offset, p.n_layers * p.hidden_dim * p.dim);
        w.w3 = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.hidden_dim);
        w.rms_final_weight = ReadFloatArray(accessor, ref offset, p.dim);
        int headSize = p.dim / p.n_heads;
        w.freq_cis_real = ReadFloatArray(accessor, ref offset, p.seq_len * headSize / 2);
        w.freq_cis_imag = ReadFloatArray(accessor, ref offset, p.seq_len * headSize / 2);

        if (sharedWeights) w.wcls = w.token_embedding_table;
    }

    private static float[] ReadFloatArray(MemoryMappedViewAccessor accessor, ref long offset, int size)
    {
        float[] array = new float[size];
        accessor.ReadArray(offset, array, 0, size);
        offset += sizeof(float) * (long) size;
        return array;
    }


    private static RunState InitializeRunState(Config cfg)
    {
        return new RunState
        {
            x = new float[cfg.dim],
            xb = new float[cfg.dim],
            xb2 = new float[cfg.dim],
            hb = new float[cfg.hidden_dim],
            hb2 = new float[cfg.hidden_dim],
            q = new float[cfg.dim],
            k = new float[cfg.dim],
            v = new float[cfg.dim],
            att = new float[cfg.n_heads * cfg.seq_len],
            logits = new float[cfg.vocab_size],
            probindex = new ProbIndex[cfg.vocab_size],
            key_cache = new float[cfg.n_layers * cfg.seq_len * cfg.dim],
            value_cache = new float[cfg.n_layers * cfg.seq_len * cfg.dim]
        };
    }


    // Transformer and RunState structs, and related memory management
    [StructLayout(LayoutKind.Sequential)]
    private struct Config
    {
        public int dim; // transformer dimension
        public int hidden_dim; // for ffn layers
        public int n_layers; // number of layers
        public int n_heads; // number of query heads
        public int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
        public int vocab_size; // vocabulary size, usually 256 (byte-level)
        public int seq_len; // max sequence length
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct TransformerWeights
    {
        // token embedding table
        public float[] token_embedding_table; // (vocab_size, dim)

        // weights for rmsnorms
        public ArraySegment<float> rms_att_weight; // (layer, dim) rmsnorm weights

        public ArraySegment<float> rms_ffn_weight; // (layer, dim)

        // weights for matmuls
        public ArraySegment<float> wq; // (layer, dim, dim)
        public ArraySegment<float> wk; // (layer, dim, dim)
        public ArraySegment<float> wv; // (layer, dim, dim)

        public ArraySegment<float> wo; // (layer, dim, dim)

        // weights for ffn
        public ArraySegment<float> w1; // (layer, hidden_dim, dim)
        public ArraySegment<float> w2; // (layer, dim, hidden_dim)

        public ArraySegment<float> w3; // (layer, hidden_dim, dim)

        // final rmsnorm
        public float[] rms_final_weight; // (dim,)

        // freq_cis for RoPE relatively positional embeddings
        public float[] freq_cis_real; // (seq_len, head_size/2)

        public float[] freq_cis_imag; // (seq_len, head_size/2)

        // (optional) classifier weights for the logits, on the last layer
        public float[] wcls;
    }

    /// <summary>
    ///     Used in top-p sampling
    /// </summary>
    private struct ProbIndex
    {
        public float Prob;
        public int Index;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RunState
    {
        // current wave of activations
        public float[] x; // activation at current time stamp (dim,)
        public float[] xb; // same, but inside a residual branch (dim,)
        public float[] xb2; // an additional buffer just for convenience (dim,)
        public float[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
        public float[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
        public float[] q; // query (dim,)
        public float[] k; // key (dim,)
        public float[] v; // value (dim,)
        public float[] att; // buffer for scores/attention values (n_heads, seq_len)
        public float[] logits; // output logits

        public ProbIndex[] probindex; // buffer used in top-p sampling

        // kv cache
        public float[] key_cache; // (layer, seq_len, dim)
        public float[] value_cache; // (layer, seq_len, dim)
    }
}