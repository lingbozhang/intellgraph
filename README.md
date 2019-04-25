# IntellGraph [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Deep Learning Framework in C++, developed with Just-in-time Compilation and Symbolic Computation.

<p style="text-align:center;"><img src="doc/incubation.png" alt="drawing"  width="200"/>

# Description
IntellGraph is an abbreviation of Intelligent Graph. As the name indicates, the IntellGraph framework is developed for Artifical Intelligence and is abstracted 
based on Graph Theory. The project is still under development. In current version, users are able to use it for constructing fully connected deep neural networks 
with different activation and loss functions (e.g. sigmoid activation function, mean square error loss function, cross-entropy loss function, etc). Examples 
(in the example/ directory) are prepared to show the capability of the IntellGraph project and you are encouraged to study them before building your own neural 
networks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
* [Homebrew](https://brew.sh) (A package manager for MacOS and could be used to install CMake and Conan)
* [CMake](https://cmake.org)
* [Conan](https://conan.io)

### Building

#### Git Clone

First we need to check out the git repo:

```bash
$ cd ${insert your workspace folder here}
$ git clone https://github.com/lingbozhang/IntellGraph my-project
```

Now we should be in the project's top level folder. 

#### Building Manually

```bash
$ rm -rf build/manual && mkdir -p build/manual
$ cd build/manual
$ conan install ../..
$ cmake ../..
$ make && make install
```
## Run examples
To run examples (codes are located in the examples/ directory), do following:
```
$ cd intellgraph/build/manual/bin
$ ./examples
```

## Contribution guidelines
We welcome every people who interested in the IntellGraph project, if you want to contribute, please do not hesitate to contact me lingboz2015@gmail.com. You are recommended to review the [contribution guidelines](CONTRIBUTING.md) and [development tutorial](DEVELOPMENT_TUTORIAL.md). By participating, you are also expected to join the Gitter community:

[![Gitter chat](https://badges.gitter.im/lingbozhang/gitter.png)](https://gitter.im/intellgraph/community)

## License
[Apache License](LICENSE)

## Acknowledgments

- IntellGraph uses and modifies the [cmake-project-template](https://github.com/kigster/cmake-project-template) developed by Konstantin Gredeskoul. 
