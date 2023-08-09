# Andrej Karpathy's llama2.c in one file of pure C#.
[llama2.c](https://github.com/karpathy/llama2.c) is a very simple implementation
to run inference of models with a [Llama2](https://arxiv.org/pdf/2302.13971.pdf)-like transformer-based LLM architecture.  

This is a pure C# implementation of the same thing. It is optimized for speed and very simple to understand and modify.


## Usage

Requires .net7 or higher.

1. First put the stories15M.bin file in the same directory as the executable. You can download it from [here](https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin)
2. Get tokenizer from [here](https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin) and put it in the same directory as the executable.
```
dotnet build -c Release
```

### Generate a random story
```
.\bin\Release\net7.0\llama2.cs.exe stories15M.bin
```
![WindowsTerminal_LrRTW3joph](https://github.com/trrahul/llama2.cs/assets/7353840/3b469a99-b83a-43f1-b07d-227da7b9ebe0)


### Generate a random story with a given prompt
```
.\bin\Release\net7.0\llama2.cs.exe stories15M.bin  -i "A long time ago a"
```

### TODO
 - [ ] Inference with Llama2 checkpoints
 - [ ] Use high performance C# types from .net8?
 - [ ] Add training functionality
