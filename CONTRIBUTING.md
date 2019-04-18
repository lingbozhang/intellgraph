# Contributing guidelines
Before sending your pull requests, make sure you followed this list.

- Read [contributing guidelines](CONTRIBUTING.md).
- Changes are consistent with the [Coding Style](https://github.com/lingbozhang/intellgraph/blob/master//CONTRIBUTING.md#c-coding-style). 
- Commit messages are consistent with the [Git commit message conventions](https://github.com/lingbozhang/intellgraph/blob/master//CONTRIBUTING.md#c-coding-style).


#### C++ coding style
Changes to IntellGraph C++ code should conform to
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

Use `clang-tidy` to check your C/C++ changes. (For Clion users, `clang-tidy` is automatically installed. For vscode users, 
`clang-tidy` can be found in vscode extension marketplace). To install `clang-tidy` on ubuntu:16.04, do:

```bash
apt-get install -y clang-tidy
```

In addition, since C++ language does not do very well in ownership like Rust. However, ownership
is very important, it helps programmer keeps in mind of memory safety while programming. Therefore, 
in IntellGraph, several dummy specifiers are defined and developers are asked to use them in the header. 
Note it is usually not recommended to introduce macros in the header.

* MUTE: indicates pass/return by pointer, reference or shared_ptr
* COPY: indicates pass/return by copy
* REF:	indicates pass/return by const or const reference/pointer
* MOVE:	indicates pass/return by rvalue, or unique_ptr in C++, move ownership is achieved with unique smart pointers.

In order to emphasize on the ownership exchange, in addition to the original accessor and mutator, new accessor and mutator are defined.
* ref_variable_name: returns const variable reference
* move_variable_name: set variable by move

In C++, abstract class is similar to interface in Java, in order to distinguish it with class, an interface macro is defined and should be used for abstract 
class. For interface, the public is replaced with implements macro.

In order to differentiate between class and interface, interface is named with lower case letters. All interfaces must have virtual destructor in order to 
allow memory release from interfaces

#### Coding style for other languages

* [Google Java Style Guide](https://google.github.io/styleguide/javaguide.html)
* [Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html)
* [Google Shell Style Guide](https://google.github.io/styleguide/shell.xml)
* [Google Objective-C Style Guide](https://google.github.io/styleguide/objcguide.html)


#### Git Commit Message Conventions
Currently, we follow git commit message conventions from [AngularJS Git Commit Message Conventions](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#heading=h.uyo6cb12dt6w) with some modifications. In IntellGraph project, the scope tag is 
neglected and the body of commit message must written in the [MarkDown format](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)












