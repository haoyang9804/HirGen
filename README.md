# HirGen

**HirGen** is an effective fuzzer for Deep Learning compilers. Specifically, it focuses on the high-level optimization stage.

Now HirGen can support 58 operators, including

+ binary operators: Add, Subtract, Multiply, Divide, Power, Mod,
Floor Mod, Floor Divide, Logical And, Logical Or, Logical
Xor, Bitwise And, Bitwise Or, Equal, Not Equal, Less, LessEqual,
Greater, GreaterEqual, Maximum, Minimum, Right Shift, Left
Shift.
+ unary operators: Log, Log2, Log10, Tan, Tanh, Cos, Cosh, Sin,
Sinh, Acos, Acosh, Asin, Asinh, Atan, Atanh, Exp, Erf, Sqrt,
Rsqrt, Sigmoid, Floor, Ceil, Trunc, Round, Abs, Sign,
Negative, Logical not, Bitwise not, Zeros Like, Ones Like,
Copy, isNan, isFinite, isInf.

Until now, **HirGen** has detected 21 bugs, of which 17 have been confirmed
and 12 have been fixed. All these bugs lie in https://github.com/anonymousWork000/HirGen/tree/experiment

To execute HirGen, first create a `build` folder, then `cmake .. -G Ninja` in `build`, finally `ninja`.
You will find `hirgen` in `build`, and just run it with `./hirgen`. To assure cmake run successfully, please specify your Clang++/Clang path in `CMakeLists.txt`. If you prefer GCC/G++ and your default C/C++ compiler is GCC/G++, you can just delete the path specification of C/C++ compiler in `CMakeLists.txt`.

`hirgen` provides several options for generating computational graphs.
  + `-num` specifies the number of operators, the default value is `-num=100`
  + `-testing` specifies whether perform test oracle 3 in paper, the default value is `-testing=nodf`, meaning no test oracle 3 included. In this choice, the testing process would be much faster but may miss bugs of calculation difference on difference hardware incurred by the running of model. If you want to enable test oracle 3, please run `hirgen` with `-testing=df`.
  + `-clevel` specifies the generation mode. Default option is `-clevel=strict`, meaning strict generation. You can switch it to disruptive generation by `-clevel=disruptive`.
  + -coverage` specifies whether we enable coverage guidance. The default option is `-coverage=yes`, you can switch it to `-coverage=no` to turn off guidance and make hirgen generate computational graph randomly.

As for TVM, please refer to [here](https://tvm.apache.org/docs/install/from_source.html) to install it from source.
After installing, you can `git checkout 124813f` to get the TVM version that can reproduce all our found bugs and then build TVM.
