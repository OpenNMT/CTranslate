# CTranslate

CTranslate is a C++ implementation of OpenNMT's `translate.lua` script. It has few dependencies and facilitates the use of OpenNMT models in existing products and on various platforms. It uses [Eigen](http://eigen.tuxfamily.org) as a backend.

It only supports CPU translation of OpenNMT models that were released with the [`release_model.lua`](https://github.com/OpenNMT/OpenNMT/tree/master/tools#release-model).

## Dependencies

* *C++11*
* *CMake*
* *Boost*
* *Eigen* > 3.3

## Compiling

```
$ mkdir build
$ cd build
$ cmake -DEIGEN_ROOT=<path to Eigen library> ..
```

It will produce the dynamic library `libonmt.so` and the translation client `cli/translate`.

To compile only the library, use the `-DLIB_ONLY=ON` flag.

## Using

### Client

See `cli/translate --help`. It has the same interface as the `translate.lua` script in OpenNMT while adding the ability to work with the standard input and output.

### Library

This project is also a convenient way to load OpenNMT models and translate texts in existing software.

Here is a very simple example:

```cpp
#include <iostream>

#include <onmt/TranslatorFactory.h>

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
* `include/onmt/ITranslator.h` to translate sequences or batch of sequences
* `include/onmt/TranslationResult.h` to retrieve results and attention vectors

## Missing features

* OpenNMT tokenization and detokenization
