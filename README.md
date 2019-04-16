# IntellGraph [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Deep Learning Framework developed in C++ and LLVM, focused on Just-in-time Compilation and Symbolic Computation

# Description
IntellGraph is an abbreviation of Intelligent Graph. As the name indicates, the IntellGraph framework is developed for Artifical Intelligence and is abstracted 
based on Graph Theory. The project is still under development. In current version, users are able to use it for constructing fully connected deep neural networks 
with different activation and loss functions (e.g. sigmoid activation function, mean square error loss function, cross-entropy loss function, etc). Some examples 
(in the example/ directory) are prepared to show the capability of the IntellGraph project and you are encouraged to study them before building your own neural 
networks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
* [Homebrew](https://brew.sh)
* [CMake](https://cmake.org)

### Building

#### Git Clone

First we need to check out the git repo:

```bash
$ cd ${insert your workspace folder here}
$ git clone https://github.com/lingbozhang/IntellGraph my-project
$ cd my-project
$ git submodule init && git submodule update
```

Now we should be in the project's top level folder. 

#### Building Manually

```bash
$ rm -rf build/manual && mkdir build/manual
$ cd build/manual
$ cmake ../..
$ make && make install
$ cd ../..

# Run the binary:
$ bin/IntellGraph
```
####  Building Using the Script

There is a handy BASH script (used by the Travis CI) that you can run locally. It builds the project, and runs main function. Note run.sh takes 2-5 minutes to setup at first run (please be patient)

```bash
./run.sh
```
## Contribution guidelines
If you want to contribute to IntellGraph, be sure to review the [contribution guidelines](CONTRIBUTING.md). By participating, you are expected to uphold this code.

## License
[Apache License](LICENSE)

## Acknowledgments

- IntellGraph uses and modifies the [cmake-project-template](https://github.com/kigster/cmake-project-template) developed by Konstantin Gredeskoul. 
