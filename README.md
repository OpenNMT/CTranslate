# CTranslate

CTranslate is a C++ implementation of OpenNMT's `translate.lua` script, with no Torch/Lua dependencies. It facilitates the use of OpenNMT models in existing products and on various platforms using [Eigen](http://eigen.tuxfamily.org) as a backend.

It supports CPU OpenNMT models. You can convert GPU trained models with the [`release_model.lua`](https://github.com/OpenNMT/OpenNMT/tree/master/tools#release-model) script.

## Dependencies

* `Boost`
* `Eigen` > 3.3

## Compiling

*CMake and a compiler that supports the C++11 standard are required to compile the project.*

*Instructions below are given for a Linux system. On MacOS, generated library will be a .dylib, on Windows a .dll.*

```
mkdir build
cd build
cmake -DEIGEN_ROOT=<path to Eigen library> -DCMAKE_BUILD_TYPE=<Release or Debug> ..
make
```

It will produce the dynamic library `libonmt.so` and the translation client `cli/translate`. To compile only the library, use the `-DLIB_ONLY=ON` flag.

By default, if the compiler is compatible, compilation is done using [OpenMP](http://www.openmp.org). To disable OpenMP, use the `-DWITH_OPENMP=OFF` flag.

### Performance tips

* Unless you are cross-compiling for a different architecture, you can add `-DCMAKE_CXX_FLAGS="-march=native"` to the `cmake` command above to optimize for speed.
* Consider using [Intel® MKL](https://software.intel.com/en-us/intel-mkl) if available. You should follow [Eigen instructions](https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html) to link against it.

## Using

### Client

See `cli/translate --help`. It has the same interface as the `translate.lua` script in OpenNMT while adding the ability to work with the standard input and output.

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
* `include/onmt/Threads.h` to programmatically control the number of threads to use

## Missing features

* OpenNMT tokenization and detokenization
