## [Unreleased]

### Breaking changes

* Translation results now have an additional dimension covering the multiple hypotheses for each batch

### New features

* Add n-best feature

### Fixes and improvements

## [v0.6.10](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.10) (2018-09-10)

### Fixes and improvements

* Allow linking to external OpenNMT/Tokenizer

## [v0.6.9](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.9) (2018-09-07)

### Fixes and improvements

* Update Tokenizer to v1.8.1

## [v0.6.8](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.8) (2018-09-07)

### Fixes and improvements

* Update Tokenizer to v1.8.0

## [v0.6.7](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.7) (2018-09-04)

### Fixes and improvements

* Update Tokenizer to v1.7.0

## [v0.6.6](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.6) (2018-08-29)

### Fixes and improvements

* Update Tokenizer to v1.6.2

## [v0.6.5](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.5) (2018-07-30)

### Fixes and improvements

* Update Tokenizer to v1.6.0

## [v0.6.4](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.4) (2018-07-13)

### Fixes and improvements

* Update Tokenizer to v1.5.3

## [v0.6.3](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.3) (2018-07-12)

### Fixes and improvements

* Update Tokenizer to v1.5.2

## [v0.6.2](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.2) (2018-07-12)

### Fixes and improvements

* Update Tokenizer to v1.5.1

## [v0.6.1](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.1) (2018-07-05)

### Fixes and improvements

* Update Tokenizer to v1.5.0

## [v0.6.0](https://github.com/OpenNMT/CTranslate/releases/tag/v0.6.0) (2018-06-14)

### New features

* Add 16 bits quantization with SSE and AVX2 optimizations
* Introduce vocabulary mapping to speed-up decoding-up
* Report profiles per modules block (encoder_fwd, encoder_bwd, decoder, generator)
* Support cloning a `Translator` while sharing the model data
* Translate batches in parallel via `cli/translate`

### Fixes and improvements

* Fix GRU support
* Update Tokenizer to v1.4.0

## [v0.5.4](https://github.com/OpenNMT/CTranslate/releases/tag/v0.5.4) (2018-04-10)

### Fixes and improvements

* Update Tokenizer to v1.3.0

## [v0.5.3](https://github.com/OpenNMT/CTranslate/releases/tag/v0.5.3) (2018-03-28)

### Fixes and improvements

* Update Tokenizer to v1.2.0

## [v0.5.2](https://github.com/OpenNMT/CTranslate/releases/tag/v0.5.2) (2018-01-23)

### Fixes and improvements

* Update Tokenizer to v1.1.1

## [v0.5.1](https://github.com/OpenNMT/CTranslate/releases/tag/v0.5.1) (2018-01-22)

### Fixes and improvements

* Update Tokenizer to v1.1.0

## [v0.5.0](https://github.com/OpenNMT/CTranslate/releases/tag/v0.5.0) (2017-12-11)

### New features

* Add module profiling
* Link against Intel® MKL if available
* [*experimental*] Offload matrix multiplication to the GPU

### Fixes and improvements

* Improve Eigen library finder logic

## [v0.4.1](https://github.com/OpenNMT/CTranslate/releases/tag/v0.4.1) (2017-03-08)

### Fixes and improvements

* Fix install rule for TH dependency

## [v0.4.0](https://github.com/OpenNMT/CTranslate/releases/tag/v0.4.0) (2017-03-08)

### New features

* Add CMake install rule
* Add static library compilation support

### Fixes and improvements

* Tokenization is now an external library

## [v0.3.2](https://github.com/OpenNMT/CTranslate/releases/tag/v0.3.2) (2017-02-08)

### Fixes and improvements

* Fix error when decoded sequences reached `max_sent_length`
* Fix incorrect extraction of word features

## [v0.3.1](https://github.com/OpenNMT/CTranslate/releases/tag/v0.3.1) (2017-01-29)

### Fixes and improvements

* Fix `--joiner_new` option when using BPE
* Fix segmentation fault when a translator is destroyed and other instances are in use

## [v0.3.0](https://github.com/OpenNMT/CTranslate/releases/tag/v0.3.0) (2017-01-26)

### New features

* Tokenization and detokenization

### Fixes and improvements

* Fix errors when using models with word features
* Remove Boost dependency when compiling as a library

## [v0.2.0](https://github.com/OpenNMT/CTranslate/releases/tag/v0.2.0) (2017-01-19)

### New features

* Programmatically set the number of threads to use

### Fixes and improvements

* Simplify project include with a single public header `onmt/onmt.h`
* Fix compilation on Mac OS

## [v0.1.0](https://github.com/OpenNMT/CTranslate/releases/tag/v0.1.0) (2017-01-11)

Initial release.
