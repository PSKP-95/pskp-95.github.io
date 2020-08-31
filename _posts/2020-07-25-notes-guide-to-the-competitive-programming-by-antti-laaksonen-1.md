---
layout: post
title: Notes - Guide to Competitive Programming (Chapter 1 - Introduction)
description: Notes of Book Guide to Competitive Programming authored by Antti Laaksonen
keywords: competitive programming, antti laaksonen
author: Parikshit Patil
thumbnail: 
---

Competitive programming combines two topics:

1. **Design of Algorithms**: The core of competitive programming is about inventingefficient algorithms that solve well-defined computational problems. The design ofalgorithms requires problem solving and mathematical skills.
2. **Implementation of Algorithms**: In competitive programming, the solutions to prob-lems are evaluated by testing an implemented algorithm using a set of test cases.Thus, after coming up with an algorithm that solves the problem, the next step is tocorrectly implement it, which requires good programming skills.

## Collatz conjecture

Consider an algorithm that takes as input a positive integern.If \\(n\\) is even, the algorithm divides it by two, and if \\(n\\) is odd, the algorithm multiplies it by three and adds one. The algorithm repeats this, until \\(n\\) is one. For example, the sequence for \\(n=3\\) is as follows:

<div class="scrollable">
<notextile>
\[
3→10→5→16→8→4→2→1
\]
</notextile>
</div>


Your task is to simulate the execution of the algorithm for a given value of \\(n\\).

**Input**

The only input line contains an integer \\(n\\).

**Output**

Print a line that contains all values of \\(n\\) during the algorithm.

**Constraints**
- \\(1≤n≤10^6\\)

**Example**

*Input:*

3

*Output:*

3 10 5 16 8 4 2 1

### Solution

```cpp
#include<iostream>
using namespacestd;
int main() {
    long long n;
    cin >> n;
    while(true){
        cout << n << " ";
        if(n == 1)
            break;
        if(n%2 == 0) 
            n /= 2;
        else
            n = n*3+1;
    }
    cout << "\n";
}
```