# IntellGraph

A Deep Learning Framework in C++


# Installation
### Building The Project

#### Git Clone

First we need to check out the git repo:

```bash
$ cd ${insert your workspace folder here}
$ git clone https://github.com/lingbozhang/IntellGraph my-project
$ cd my-project
$ git submodule init && git submodule update
```

Now we should be in the project's top level folder. 

#### Project Structure

There are three empty folders: `lib`, `bin`, and `include`. Those are populated by `make install`.

`src` is the sources, and `test` is where we put our unit tests.

Now we can build this project, and below we show two separate ways to do so.

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

### Acknowledgements
We use the cmake-project-template developed by Konstantin Gredeskoul. For more information, please refer to https://github.com/kigster/cmake-project-template

### Useful links
[AngularJS Git Commit Message] Conventions(https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#heading=h.uyo6cb12dt6w)
