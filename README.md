[![Build Status](https://api.travis-ci.org/OpenNMT/CTranslate.svg?branch=master)](https://travis-ci.org/OpenNMT/CTranslate)

# CTranslate

CTranslate is a C++ implementation of OpenNMT's `translate.lua` script with no LuaTorch dependencies. It facilitates the use of OpenNMT models in existing products and on various platforms using [Eigen](http://eigen.tuxfamily.org) as a backend.

It only supports CPU translation of OpenNMT models released with the [`release_model.lua`](https://github.com/OpenNMT/OpenNMT/tree/master/tools#release-model) script.

## Dependencies

* `Eigen` > 3.3

Compiling executables additionally requires:

* `Boost` (`program_options`)

## Compiling

*CMake and a compiler that supports the C++11 standard are required to compile the project.*

```
mkdir build
cd build
cmake -DEIGEN_ROOT=<path to Eigen library> -DCMAKE_BUILD_TYPE=<Release or Debug> ..
make
```

It will produce the dynamic library `libonmt.so` (or `.dylib` on Mac OS, `.dll` on Windows), the translation client `cli/translate` and tokenization tools `cli/tokenize` and `cli/detokenize`.

### Options

* To compile only the library, use the `-DLIB_ONLY=ON` flag.
* To disable [OpenMP](http://www.openmp.org), use the `-DWITH_OPENMP=OFF` flag.

### Performance tips

* Compile in release mode (`-DCMAKE_BUILD_TYPE=Release`)
* Unless you are cross-compiling for a different architecture, add `-DCMAKE_CXX_FLAGS="-march=native"` to the `cmake` command above to optimize for speed.
* Consider using [Intel® MKL](https://software.intel.com/en-us/intel-mkl) if available. You should follow [Eigen instructions](https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html) to link against it.

## Using

### Clients

See `--help` on the clients to discover available options and usage. They have the same interface as their Lua counterpart.

### Library

This project is also a convenient way to load OpenNMT models and translate texts in existing software.

Here is a very simple example:

```cpp
#include <iostream>

#include <onmt/onmt.h>

int main()
{
  // Create a new Translator object.
  auto translator = onmt::TranslatorFactory::build("enfr_model_release.t7");

  // Translate a tokenized sentence.
  std::cout << translator->translate("Hello world !") << std::endl;

  return 0;
}

```

For a more advanced usage, see:

* `include/onmt/TranslatorFactory.h` to instantiate a new translator
* `include/onmt/ITranslator.h` (the `Translator` interface) to translate sequences or batch of sequences
* `include/onmt/TranslationResult.h` to retrieve results and attention vectors
* `include/onmt/Tokenizer.h` to apply OpenNMT's tokenization and detokenization
* `include/onmt/Threads.h` to programmatically control the number of threads to use
