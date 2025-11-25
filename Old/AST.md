Abstract Syntax Tree in 5 minutes!


Add this to your development workflow for 10x smoother operations!

When writing code, the flow is conventionally like this:

Write code design, draw flowchart(s), define algorithm.
Write actual code
However, programmers like to challenge the norm and they reverse the flow subconsciously because, let’s be honest, who writes documentation first?

The other day, I was updating a piece of documentation to fit the feature code written in my game. I know, I know, I do it all the time too!

But the point is: I needed a tool to parse through the code change so I know what to update in my documentation. Of course, I could have opened up my diff tool such as BeyondCompare and check one by one which function got deleted or which line was removed as a result of some unoptimized code by freshies like me. But that would take a lot of time.

I then resorted to using Python to help parse each line of the code.

Until… it wasn’t working as efficiently as I’d expected.

You see, using Python to parse each line of code as text is dangerous as hell. The thing is, the parser doesn’t know the context of the code! You risk every line of your code to subtle changes that you might not notice. Then, I came across the concept of Abstract Syntax Tree or AST. This changed my workflow.

Abstract Syntax Tree (AST)
By textbook definition, AST is a tree representation of your source code or program. Within this tree, you get the structure of your code. If you are familiar with the concept of a tree, you know nodes. What’s powerful with this tree is that the structure encompasses the context and content of your code in one tree.

What I mean here is, for example, you want a parser to tell you which lines in your code constitute a function name. If you use Python RegEx, you would parse each line as text and find the necessary patterns that refer to a function. But, there could be many patterns and certainly, it is only a matter of time until your parser runs out of methods to find the patterns.

AST, on the other hand, knows by heart which lines are your functions, local variables, even their assigned values and any conditional-checking statements. Even the order of execution of the code is saved in the tree!

And yes, this tree is an actual representation of how a compiler works. I’ll save you the time of some boring explanation here, you can check out the reference links in the reference section for more details. I want to show a real-life use case that I find useful in my workflow.

Bonus:
If you wonder sometimes how test coverage tools work, AST comes into the picture actually. With ASTs, the test coverage tool injects additional code into your source code to keep track of the coverage of each line. Then, at the end of the test, the tool uses AST to give you a proper insight of your test execution. Imagine doing this using a Python script. Insane.

What to do with ASTs?
Sure, our goal is not to build another compiler or test coverage tool like CTC or Istanbul, but if you can see what the tree brings us, there’s actually a lot that we can do.

One of the useful things I find with AST is that it can help generate test cases for my code.

Again, I know. If you are telling me this isn’t the right “flow” for test-driven development (TDD), I know! But I think if your goal is to write good code, as long as the tool helps in ensuring the code runs perfectly fine, whichever step comes first should be alright, no?

Actual workflow of Test-Driven Development:

Write test cases for how you think the feature should be tested.
Write source code to fit the test.
So, bear with me and imagine this workflow:

You write your feature code. You think your code works fine because you want to follow the Behavior-Driven Development.
Now you write test cases to test the behavior.
Now, you are at the point of writing test cases based on your source code. Sure, you can write them one by one to confirm if the outputs are as expected.

What if… there is a tool that can help you generate test cases? And all you need to do is to specify your outputs according to each test case.

This is where AST comes in. Let me show you.

Peeping at the tree
With a simple command, you can take a look at the tree content:

clang -Xclang -ast-dump -fsyntax-only test.c
You can also apply filter to a different command to check the content of the tree.

clang-check -ast-dump -ast-dump-filter=func test.c
A sample of what it may look like:

Press enter or click to view image in full size

Using AST to generate test cases
Preconditions
Install LLVM here.
In Python, pip install clang.
C Source code
#include "testfunc.h"
#include "stdio.h"
#include "math.h"

int func(int x, int y)
{
    int a = 0;
    int b = 10;

    a = pow(x, 2) + y*3 + 6;
    if (a > b)
    {
        b = 0;
    }
    else
    {
        b = 20;
    }

    return b;
}

int main()
{
    int result = func(2, 3);
    return 0;
}
A very simple piece of C code above will be our file to parse.

Python Test Case Generation Tool
Using Python, we can reference our C source code and generate its structural contents.

Let’s setup our file to test in our Python test generation tool.

# File setup
file_to_test = "testfunc.c"
extracted_file_to_test = file_to_test.split(".")[0]
test_filename = "../test/unittest_" + extracted_file_to_test + ".cpp"
f = open(test_filename, "w")
Let’s not forget our AST parser tool setup.

# AST parser setup
import clang.cindex
import os
os.environ["LLVM_LIB_PATH"] = "C:\\Program Files\\LLVM\\bin"
if os.getenv('LLVM_LIB_PATH') is None:
    print("ERROR:", "LLVM_LIB_PATH environment variable is unset")
    raise SystemExit

clang.cindex.Config.set_library_path(os.getenv('LLVM_LIB_PATH'))
We can start parsing the tree like this:

parser_args = ['-std=c99']
index = clang.cindex.Index.create()
tu = index.parse('../testfunc.c',
                 options=clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD |
                 clang.cindex.TranslationUnit.PARSE_INCOMPLETE,
                 args=parser_args)
From here onwards, the only thing we need to do is to traverse the tree and save everything we need to save into our own data structure format.

def traverse(node):
    func_nodes = []
    param_nodes = []
    param_type_nodes = []
    return_nodes = []

    if (str(node.location.file).endswith(".h")):
        # do nothing
        return 1
    else:
        if (node.kind == clang.cindex.CursorKind.FUNCTION_DECL):
            func_nodes.append(node.spelling)

            for n in node.get_children():
                if (n.kind == clang.cindex.CursorKind.PARM_DECL):
                    param_nodes.append(n.spelling)
                    param_type_nodes.append(n.type.spelling)

                if (n.kind == clang.cindex.CursorKind.RETURN_STMT):
                    return_nodes.append(node.spelling)
        
        return func_nodes, param_nodes, param_type_nodes, return_nodes
Start parsing and store our data!

root = tu.cursor
for n in root.get_children():
    retval = traverse(n)
    if (retval != 1):
        function_nodes, parameter_nodes, parameter_type_nodes, return_nodes = retval
    global_func_nodes.append(function_nodes)
    global_param_nodes.append(parameter_nodes)
    global_param_type_nodes.append(parameter_type_nodes)
    global_return_nodes.append(return_nodes)
To parse through our own data structure to generate the test cases, I quickly coded up an unoptimized algorithm to do this. Of course, you can clean up the code and write better functions for the necessary tasks.

for idx, function in enumerate(global_func_nodes):
    if function != []:
        f.write("TEST_F(" + extracted_file_to_test.capitalize() + "_Test, " + function[0] + ")")
        f.write("\n")
        f.write("{")
        f.write("\n")
        if len(global_param_nodes[idx]) != 0:
            for param_idx, param in enumerate(global_param_nodes[idx]):
                f.write("\t")
                f.write(global_param_type_nodes[idx][param_idx] + " ")
                f.write(global_param_nodes[idx][param_idx] + ";\n")

            if global_return_nodes[idx] is not None:
                f.write("\t")
                f.write(global_param_type_nodes[idx][param_idx] + " ")
                f.write("retVal = ")
            else:
                f.write("\t")
            f.write(function[0] + "(")
            for param_idx, param in enumerate(global_param_nodes[idx]):
                f.write(global_param_nodes[idx][param_idx])
                if param_idx != len(global_param_nodes[idx]) - 1:
                    f.write(", ")
                else:
                    f.write(");")
                    f.write("\n")
                
            f.write("}")
            f.write("\n\n")
        else:
            f.write("\n")
            f.write("}")
            f.write("\n\n")
And with this, the generated test cases are stored in another file and they look fine to me!

TEST_F(Testfunc_Test, func)
{
   int x;
   int y;
   int retVal = func(x, y);
}

TEST_F(Testfunc_Test, main)
{

}
Except that I forgot to insert the necessary headers in order to run Googletest. But if you read my article on how to setup and run a Googletest in this last article, I’m pretty sure you will have it under control!

Now, you have generated some working and presumably passing test cases by just using the AST tool! Imagine all the time that you can save by not spending on writing test cases. Let the tool generate them and you focus on the test results.

However, back to the topic of test-driven development, this is only useful when you are running out of time during the development phase. If you are concerned with the code quality, take a good amount of time and craft your tests before writing your code.

I hope I have done a good job at showing how to use the AST tool to parse our C code, which can then help us generate test cases in a split second if we use it correctly.

Read my other articles on breaking down tough and baffling technical topics.