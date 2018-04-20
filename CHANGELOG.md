## [Unreleased]

### New features

* Introduce vocabulary mapping to speed-up decoding-up
* Report profiles per modules block (encoder_fwd, encoder_bwd, decoder, generator)

### Fixes and improvements

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
