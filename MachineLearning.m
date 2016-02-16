[Machine Learning]

# Machine Learning is the science of getting computers to learn, without being explicitly programmed

-- Additional Material

- https://share.coursera.org/wiki/index.php/ML:Main

-- Week 1

INTRODUCTION

[Supervised Learning]

- Supervised Learning --> 'right answers' are given
- Regression --> predict continuous valued output (price) # housing price example
- Classification --> discrete valued output (0 or 1; there can be more discrete categories than 2) # breast cancer tumor example

[Unsupervised Learning]

- All data has the same label, no right or wrong
- Here is the dataset, can you find some structure?
  + Clustering Algorithms
  + Cocktail party algorithm # [W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');

LINEAR REGRESSION WITH ONE VARIABLE

m = number of training examples
x's = 'input' variable / features
y's = 'output' variable / 'target' variable
(x, y) = a single training example
(x^(i), y^(i)) = refers to the i'th training example --> index, no exponantiation!

ML Flow:
1 - Training Set --> Learning Algorithm --> h (hypothesis; in the form of a function)
1.1 - Size of House (x) --> h --> Estimated Price (y) # h maps from x's to y's

How do we represent h?
- h0(x) = θ0 + θ1x # shorthand: h(x)
- Linear regression with one variable / univariate linear regression.

θi's = parameters

How to choose θi's?
- Idea: Choose θ0 & θ1 so that hθ(x) is close to y for our training examples (x, y)
- 1 / '2m' * Σ(h0(x^(i)) - y^(i))^2 (m Σ i = 1) is as small as possible (the distance of the data points to the line)
- Squared error function

Hypothesis: hθ(x) = θ0 + θ1x
Parameters: θ0, θ1
Cost Function: J(θ0, θ1) = 1 / '2m' * Σ(h0(x^(i)) - y^(i))^2
Goal: minimize J(θ0, θ1) # Simplified: minimize J(θ1)
       θ0, θ1

You basically plot several lines using various values for θ1 and use the J(θ1) function to determine the distances of the square of the actual values.
You select the value of θ1 where the distance is as little as possible.

- Contour plots/figures

[Gradient Descent]

- Algorithm for minimizing cost function J (see above)
- Outline:
  + Start with some θ0, θ1 (or θ0, θ1, ... θn) --> usually you initiate them at 0
  + Keep changing θ0, θ1 to reduce J(θ0, θ1) until we hopefully end up at a minimum

repeat until convergence {
	θj := θj - α*((∂/∂θj) * J(θ0, θ1)) (for j = 0 and j = 1, our 2 θ's)
}

α = learning rate, how big the step is we take each time
((∂/∂θj) * J(θ0, θ1)) = derivative term

∂ = partial derivative (when you have more than 1 parameter)
d = derivative (1 parameter)

a := b --> assignment
a = b  --> truth assertion

Simultaneous Update

Correct
+ temp0 := θ0 - α*((∂/∂θ0) * J(θ0, θ1))
+ temp1 := θ1 - α*((∂/∂θ1) * J(θ0, θ1))
+ θ0 = temp0
+ θ1 = temp1

Incorrect
- temp0 := θ0 - α*((∂/∂θ0) * J(θ0, θ1))
- θ0 = temp0
- temp1 := θ1 - α*((∂/∂θ1) * J(θ0, θ1))
- θ1 = temp1

Derivative Term

- What is the slope (height/width) of a line that just tangent to the point where we are?

Simplified (1 parameter) # in our example the slope is positive

J(θ1)
θ1 := θ1 - α*((∂/∂θ1) * J(θ1))
θ1 := θ1 - α * (some positive number) # hence we go to the left (negative)

If α is too small, gradient descent can be slow.
If α is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge.

Gradient descent can converge to a local minimum, even with th learning rate α fixed
As we approach a local minimum, gradient descent will automatically take smaller steps (since the slope becomes smaller with every step towards the minimum).
So no need to decrease α over time.

Gradient descent can optimise any cost function, but we will use it now to optimise our linear regression cost function (squared error cost function).

((∂/∂θj) * J(θ0, θ1)) = ((∂/∂θj) * 1 / '2m' * Σ(h0(x^(i)) - y^(i))^2 = ((∂/∂θj) * 1 / '2m' * Σ(θ0, θ1x^(i) - y^(i))^2

^(i) != exponantiation, simply indexing!
http://math.stackexchange.com/questions/70728/partial-derivative-in-gradient-descent-for-two-variables

Gradient descent algorithm

repeat until convergence {

	θ0 := θ0 - α * 1/m * Σ(hθ(x^(i)) - y^(i)) # ∂/∂θ0 J(θ0,θ1)
	θ1 := θ1 - α * 1/m * Σ(hθ(x^(i)) - y^(i)) * x^(i) # ∂/∂θ1 J(θ0,θ1)

} # update θ0 & θ1 simultaneously!

Since the regression cost function is a 'bow' shaped one (a 'Convex function') there is only one optimal spot in this case
(since gradient descent can lead to differing outcomes depending on the starting position, when there are differing local optimums).

'Batch' Gradient Descent

'Batch': Each step of gradient descent uses all the training examples (sometimes people prefer to only use subsets of the training data).

There is an alternative to gradient descent for determining the optimum, via solving an equation, which is called the 'normal equations method'.
Gradient descent scales better for large datasets.

[Linear Regression with Multiple Variables] # Multivariate linear regression

Notation
n = number of features
m = number of rows/training examples
x^(i) = input (features) of i'th training example (in case of multiple features this is a vector; an n dimensional vector)
xj^(i) = value of feature j in i'th training example

Hypothesis
- Previously: hθ(x) = θ0 + θ1x
- Now: hθ(x) = θ0 + θ1x1 + θ2x2 + θ3x3 + θ4x4 ... + θnxn

For convenience of notation, define x0 = 1 (x0^(i) = 1)

x = [x0, x1, x2, ..., xn] # n+1 dimensional vector, indexed from 0
θ = [θ0, θ1, θ2, ..., θn] # another n+1 zero-indexed vector

Hence:
hθ(x) = θ0x0 + θ1x1 + ... + θnxn == θ0 + θ1x1 + θ2x2 + θ3x3 + θ4x4 ... + θnxn # θ0x0 == θ0 since x0 = 1

hθ(x) = θ^Tx # transpose θ * x --> you transpose so you can use vector multiplication

[θ0, θ1, θ2, ..., θn] * [ x0
						  x1
						  ...
						  xn ] # etc.

-- Gradient descent for multiple variables

Hypothesis: θ^T * x
Parameters = (θ0, θ1, ..., θn) = θ # n+1-dimensional vector
Cost function = J(θ) = 1 / '2m' * Σ(h0(x^(i)) - y^(i))^2
Gradient descent:

New algorithm (n >= 1):

Repeat {
	θj := θj - α * 1/m * Σ(hθ(x^(i)) - y^(i)) * xj^(i)
} # simultaneously θj

And remember: x0^(i) = 1, so even if now you carry something down from the differential (because we introduced x0, it is no longer a constant), we simply multiply by 1. It is the same as == θ0 := θ0 - α * 1/m * Σ(hθ(x^(i)) - y^(i))

-- Gradient descent in practice I: Feature scaling --> make gradient run much faster and converge sooner

Feature Scaling (divide number by the maximum value)
- Idea: make sure features are on a similar scale.

Example:
x1 = size (0-2000 feet^2)
x2 = number of bedrooms (1-5)

Scaled:
x1 = size (feet^2) / 2000 # 0 <= x1 <= 1
x2 = number of bedrooms / 5 # 0 <= x2 <= 1

Get every feature into approximately a -1 <= xi <= 1 range; not much bigger or smaller.
This helps gradient descent find an optimum fast, if ranges differ a lot it may take very long.

In addition to scaling: Mean Normalization

- Replace x^i with x^i - μ to make features have approximately zero mean (do not apply to x^0 = 1)

Example:
x1 = (size-1000) / 2000
x2 = #bedrooms-2 / 5

Then (approximately): -0.5 <= x1 <= 0.5, -0.5 <= x2 <= 0.5

x1 <- x1 - μ1 / s # s = std or (max - min)

-- Gradient descent in practice II: Learning Rate

θj := θj - α * ∂/∂θj of J(θ)

- Debugging: how to make sure gradient descent is working correctly.
  + Plot the value of J(θ) on the Y-axis, the number of iterations on the X-axis: the curve should go down (with more iterations you get to a lower end value; i.e. optimum)
  + Where the slope becomes flat first, you have reached convergence (at that number of iterations).
  + Alternative: declare convergence if J(θ) decreases by less than 10^-3 (example; arbitrary number epsilon) in one iteration # example of an automatic convergence test
- How to choose learing rate α.
  + If the slope of the plot (described above) is actually increasing (when it overshoots the minimum), use a smaller α.
  + Similar if the slope goes up/down/up/down, etc., use smaller α.

- For sufficiently small α, J(θ) should decrease on every iteration.
- But if α is too small, gradient descent can be slow to converge.

Summary:
- If α is too small: slow convergence.
- If α is too large: J(θ) may not decrease on every iteration; may not converge.
- Plot J(θ) over the # of iterations to see what is going on.

To choose α, try a range of values: 0.001, 0.01, 0.1, 1
Better: 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.1 --> every increase is about '3X'

-- Features and Polynomial Regression

Housing Prices Prediction

Creating/defining new features:
1 - hθ(x) = θ0 + θ1 * frontage + θ2 * depth
2 - Land Area --> x = frontage * depth
3 - hθ(x) = θ0 + θ1x # x = land area, which uses/combines the two prior features

Polynomial Regression

Quadratic function --> θ0 + θ1x + θ2x^2 # goes up and goes back down (TODO: check why quadratic function go down again?)
Cubic function --> θ0 + θ1x + θ2x^2 + θ3x^3
Root function --> θ0 + θ1x + θ2x^0.5 # never slopes down again

Therefore having insights into how functions shape lines can be useful to choose the correct model.

hθ(x) = θ0 + θ1x1 + θ2x2 + θ3x3
      = θ0 + θ1(size) + θ2(size)^2 + θ3(size)^3

x1 = (size)
x2 = (size)^2
x3 = (size)^3

Size:   1 - 1,000
Size^2: 1 - 1,000,000
Size^3: 1 - 10^9

--> Therefore scaling becomes important!

[Normal Equation] % method to solve for θ analytically

Intuition: If 1 D (θ is real value / scalar, not a vector of parameters):
J(θ) = aθ^2 + bθ + c
∂/∂θ J(θ) = 0 % to get the minimum set the derivative of J(θ) to 0 (zero)

In the case when θ is a n+1 dimensional vector, you take the partial derivates of all θ's and set those to 0. Finally you can combine those values to get the final result.

X = matrix with x0 ... xn % m*(n+1) dimensional matrix
y = m-dimensional vector

θ = (X^T * X)^-1 * X^T * y

(X^T*X)^-1 is the inverse of matrix X^T*X
Octave: pinv(X'*X)*X'*y % X' == X^T

When using this method:
- Feature scaling is unnecessary, large variations in range are o.k.

When to use which: % m training examples, n features

A - Gradient Descent
- Need to choose α
- Needs many iterations
+ Works well even when n is large

B - Normal Equation
+ No need to choose α
+ Don't need to iterate
- Need to compute (X^TX)^-1 % this is an n*n matrix
  - Slow if n is very large

n = 100 	% NE
n = 1000	% NE
n = 10000 	% GD / NE
n > 10000   % GD

[Octave Tutorial]

- Octave is a good prototyping language. Large scale implementation can follow in a different language.

-- [1] Basic Operations

% is comment
Basic math operators: + - * / ^
Logical - equality: ==
Logical - Non-Equality: ~=
&& = AND % 1 && 0 = 0
|| = OR % 1 || 0 = 1
xor(1,0) = 1
PS1('>> '): % use to change the prompt to '>> '.
>> Variable assignment: a = 3
>> a = 3
>> a = 3; % semicolon suppresses print output
>> b = 'hi'
>> c = (3>=1);
>> c
>> c = 1 % evaluates to true
>> a = pi;
>> a
>> 3.1416
>> disp(a) % display a
>> 3.1416
>> disp(sprintf('2 decmals: %0.2f', a)) % sprintf creates a string, substitute %0.2f with a using only 2 decimals
>> 2 decimals: 3.14
>> disp(sprintf('6 decimals: %0.6f', a))
>> 6 decimals: 3.141593
>> format long % causes numbers to be displayed with many decimal numbers
>> a
>> a = 3.141592665358979
>> format short % restores the default shorter version
>> a
>> a = 3.1416
>> A = [1 2; 3 4; 5 6] % semicolon means 'go to the next row'
>> A = % 3x2 matrix

      1  2
      3  4
      5  6

>> v = [1 2 3]
>> v = % 1x3 matrix / 3-dimensional row vector

       1 2 3 % row vector

>> v = [1; 2; 3;]
>> v = % 3x1 matrix / 3-dimensional column vector

        1
        2
        3

>> v = 1:0.1:2 % start with 1, increment with 0.1, up to 2
>> v = 1.0 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 % row vector, 1x11 matrix
>> v = 1:6
>> 1 2 3 4 5 6
>> ones(2,3)
ans =

       1  1  1
       1  1  1

>> C = 2 * ones(2,3) % == [2 2 2; 2 2 2]
>> C =

       2  2  2
       2  2  2

>> w = ones(1,3)
>> w =

      1  1  1

>> w = zeros(1,3)
>> w =

     0  0  0

>> w = rand(1,3) % gives random numbers between 0 and 1 (from the uniform distribution) --> rand(rows,cols)
>> w =

     0.91477  0.14359  0.8460

>> rand(3,3)
ans =

     0.390426  0.264057  0.683559
     0.041555  0.314703  0.506769
     0.521893  0.739979  0.387001

>> w = randn(1,3) % uses gaussian distribution, with mean zero and variance/stdev 1 --> randn(rows,cols)
>> w = -6 + sqrt(10) * (randn(1,10000));
>> hist(w) % produces historigram
>> hist(w, 50) % use more bins (50)
>> eye(4) % 4x4 identity matrix
ans =

Diagonal Matrix

    1  0  0  0
    0  1  0  0
    0  0  1  0
    0  0  0  1

>> help eye % brings up the help function for eye, etc.

-- [2] Moving Data Around

>> A = [1 2; 3 4; 5 6]
A =

    1  2
    3  4
    5  6

>> size(A) % returns a 1x2 matrix
>> ans =

    3  2

>> size(A,1)
ans = 3 % first item of size(A) --> rows
>> size(A,2) % second item of size(A) --> columns
>> v = [1 2 3 4];
>> length(v)
ans = 4
>> length(A) % it alwasy returns the length of the longest dimension (rows/columns)
ans = 3
>> pwd % shows the current directory / path that Octave is in
ans = C:\Octave\3.2.cc-4.4.0\bin
>> cd 'C:\Users\ang\Desktop' % change directory
>> pwd
ans = C:\Users\ang\Desktop
>> ls % lists files/directories in pwd
>> load featuresX.dat
>> load priceY.dat
>> load('featuresX.dat') % similar as above
>> who % shows which variables are in your Octave workspace
>> size(priceY)
ans =

    47 1

>> size(featuresX)
ans =
    47 2

>> whos % gives detailed view of files in workspace
>> clear featuresX % removes featuresX from scope
>> v = priceY(1:10) % returns a vector with the first 10 elements of priceY (1x10)
>> save hello.mat v; % save variable v as hello.mat + saves it as binary data (so very compressed)
>> clear % without variable after 'clear', this clear everything from your workspace
>> save hello.txt v -ascii % now it will save it in human-readable form, ascii (text)
>> A = [1 2; 3 4; 5 6]
A =

    1  2
    3  4
    5  6

>> A(3,2) % indexing, third row, second column
ans = 6
>> A(2,:) % everything from row 2 --> ':' means every element along that row/column
ans = 3  4
>> A(:,2)
ans =

    2
    4
    6

>> A([1,3], :)
ans =

    1  2
    5  6

>> A(:,2) = [10; 11; 12]
ans =

    1  10
    3  11
    5  12

>> A = [A, [100; 101; 102]]  % append another column vector to the right
ans =

    1  10  100
    3  11  101
    5  12  102

>> A(:) % put all elements of A into a single vector
ans = % 9x1 matrix

    1
    3
    5
    10
    11
    12
    100
    101
    102

>> A = [1 2; 3 4; 5 6];
>> B = [11 12; 13 14; 15 16];
>> C = [A B] % simply concatenates A & B adjacent; SAME AS [A, B]
C = % 3x4 matrix

    1  2  11  12
    3  4  13  14
    5  6  15  16

>> C = [A;B] % semicolon means put it at the bottom, as before
C = % 6x2 matrix

    1  2
    3  4
    5  6
    11  12
    13  14
    15  16

-- [3] Computing on Data

>> A = [1 2; 3 4; 5 6];
>> B = [11 12; 13 14; 15 16];
>> C = [1 1; 2 2];
>> A * C % normal matrix multiplication
>> A .* C % element wise multiplication
ans =
    11  24
    39  56
    75  96

% in general the period '.' tends to mean "element-wise" operations in Octave
>> A .^ 2
ans =

    1   4
    9   16
    25  36

>> v = [1; 2; 3]
>> 1 ./ v
ans =

    1.000
    0.500
    0.333

>> log(v)
ans =

    0.00000
    0.69315
    1.09861

>> exp(v) % e^1, e^2, e^3 (since v contains 1, 2 & 3)
>> abs(v) % take the absolute values of v
>> -v % gives the v * -1
>> v + ones(length(v), 1) % == v + 1
ans =

    2
    3
    4

>> A' % transpose
ans =

    1  3  5
    2  4  6

>> a = [1 15 2 0.5]
>> val = max(a)
val = 15
>> [val, ind] = max(a) % give me the value and index of max(a)
val = 15
ind = 2 % index
>> max(A) % where A is a matrix --> gives column-wise maximum
ans =

    5  6

>> a
a = 1.000 15.000 2.0000 0.5000
>> a < 3 % element wise comparison
ans =
    1  0  1  1
>> find(a < 3)
ans =

    1  3  4 % the first/third/fourth element are < 3

>> A = magic(3) % magic squars: all rows, diagnoals and columns sum up to the same thing
A =

    8  1  6
    3  5  7
    4  9  2

>> [r, c] = find(A >= 7)
r =

    1
    3
    2

c =

    1
    2
    3

>> A(2,3)
ans = 7
>> sum(a) %returns the sum of a
ans = 18.5000
>> prod(a) % returns the product of a
ans = 15
>> floor(a) % rounds down the elements of a
ans =

     1  15  2  0

>> ceil(a) % round up the elemnts of a
ans =

    1  15  2  1

>> max(rand(3), rand(3))' % will return a random 3x3 matrix, where it constantly picked the higher of two random items (so all random numbers tend to be higher)
>> max(A, [], 1) % 1 == take the max along the columns, returns a row vector / same as max(A)

ans =
    8  9  7

>> max(A, [], 2) % 2 == take the max along the rows, returns a column vector

ans =

    8
    9
    7

>> max(max(A)) % maximum value of matrix (max of the max per column)
ans = 9
>> max(A(:)) % A(:) returns all matrix values in one column vector
ans = 9
>> A = magic(9)
>> sum(A,1) % sum per columns, returns column vector
>> sum(A,2) % sum per row, returns row vector
>> A .* eye(9) % this will basically multiply all elements on the diagonal with 1, all others with 0 --> eye returns identity matrix
>> sum(sum(a .* eye(9)))
ans = 369
>> sum(sum(A.*flipud(eye(9)))) % flips the identity matrix so that we can check for the second diagonal / flip Up Down == flipud

1 0 0				 0 0 1
0 1 0 --> flipud --> 0 1 0
0 0 1 				 1 0 0

>> A = magic(3)
A =

    8  1  6
    3  5  7
    4  9  2

>> pinv(A) % pseudo-inverser, returns the inverse
>> temp = pinv(A)
>> temp * A --> returns identity matrix

-- [4] Plotting Data

>> t = [0:0.01:0.98];
>> y1 = sin(2*pi*4*t);
>> plot(t, y1);
>> y2 = cos(2*pi*4*t)
>> plot(t, y2)
>> hold on;
>> plot(t, y1, 'r'); % 'r' == red
>> xlabel('time')
>> ylabel('value')
>> legend('sin', 'cos')
>> title('my plot')
>> print -dpng 'myplot.png'
>> help plot
>> close % closes plot window
>> figure(1); plot(t,y1);
>> figure(2); plot(t,y2); % now 2 windows with plots are open
>> subplot(1,2,1); % sub-divivides plot to a 1x2 grid, access first element (1x2x1)
>> plot(t,y1);
>> subplot(1,2,2);
>> plot(t,y2);
>> axis([0.5 1 -1 1]);
>> clf; % clears figure
>> a = magic(5)
>> imagesc(A) % plots a 5x5 grid of colors
>> imagesc(A), colorbar, colormap gray;
>> imagesc(magic(15)), colorbar, colormap gray; % comma-chaining function calls --> piping
>> a=1, b=2, c=3
a = 1
b = 2
c = 3

-- [5] For, while, if statements, and functions

>> v = zeros(10,1)
>> for i = 1:10,
> v(i) = 2^i;
> end;
v =
    2
    4
    8
    16
    32
    64
    128
    256
    512
    1024

>> indices = 1:10;
indices =
    1  2  3  4  5  6  7  8  9  10

>> for i = indices,
> disp(i);
> end; % break & continue also work
>> i = 1;
>> while i <=5,
> v(i) = 100;
> i = i+1;
> end;
>> v
v =
    100
    100
    100
    100
    100
    64
    128
    256
    512
    1024

>> i = 1;
>> while true,
>    v(i) = 999;
>    i = i + 1;
>    if i == 6,
>      break;
>    end;
> end;
v =

    999
    999
    999
    999
    999
    64
    128
    256
    512
    1024

>> v(1)
ans = 999
>> v(1) = 2;
>> if v(1) == 1,
>    disp('The value is one');
>  elseif v(1) == 2;
>    disp('The value is two');
>  else
>    disp('The value is not one or two.');
>  end;
The value is two
>> exit % exits octave
>> quit % quits octave

-- Defining functions in Octave
+ Create a different text file containing function, e.g. squareThisNumber.m

function y = squareThisNumber(x)

y = x^2;

>> squareThisNumber(5)
error: 'squareThisNumber' undefined near line 18 column 1
>> pwd % check & change diretory to where file is located
>> squareThisNumber(5)
ans = 25
>> % Octave search path (advanced/optional)
>> addpath('C:\Users\ang\Desktop') % adds path to list of paths for Octave to check
>> cd 'C:\'
>> squareThisNumber(5)
ans = 25 % even now we switched directory, we can still find the function

% Within octave you can create functions that return multiple values

function[y1,y2] = squareAndCubeThisNumber(x)

y1 = x^2;
y2 = x^3;

>> [a,b] = squareAndCubeThisNumber(5);
>> a
a = 25
>> b
b = 125

% Goal: Define a function to compute the cost function J(θ)

>> X = [1 1; 1 2; 1 3]
X =
    1  1
    1  2
    1  3

>> y = [1; 2; 3]
y =
    1
    2
    3

theta = [0;1]

function J = costFunctionJ(X, y, theta)

% X is the "design" matrix containing our training examples
% y is the class labels

m = size(X,1)		   			% number of training examples, size(X,2) would count columns I think
predictions = X * theta; 		% predictions of hypothesis on all m examples
sqrErrors = (predictions-y).^2; % squared errors

J = 1/(2*m) * sum(sqrErrors);

>> j = costFunctionJ(X,y,theta)
j = 0 % since theta = [0,1] corresponds to exactly the 45 degree line

>> theta = [0;0]
>> j = costFunctionJ(X,y,theta)
j = 2.3333
>> (1^2 + 2^2 + 3^2) * (2*3)
ans = 2.3333

-- [6] Vectorization

 Assume: h(θ)x = Σ(0 j*x j) ==  θ^T * x

Unvectorized implementation

prediction = 0.0;
for j = 1:n+1,
	prediction = prediction + theta(j) * x(j)
end;

Vectorized Implementation

prediction = theta' * x

QUIZ Vectorized Implementation

θ := θ - α * ∂
where ∂ == 1/m * Σ(hθ(x^(i)-y^(i))) * x^(i) --> NOTHING GETS SQUARED HERE! You are at the derivate! Hence the (squared is gone!) 1/2m is gone --> 2 * 1/2m == 1/m!!

- both θ & ∂ are n+1 dimensional vectors

-- [7] Normal equation and non-invertibility

θ = (X^T*X)^-1 * X^T * y

- What if X^T*X is non-invertible (singular/degenerate)?
- Octave: pinv(X'*X)*X'*y

pinv: pseudo-inverse % will return the correct result even if  X^T*X is non-invertible
inv: inverse

What if X^T*X is non-invertible?
- Redundant features (linearly dependent).
  + E.g. x1 = size in feet^2
  + x2 = size in m^2
    % lm = 3.28 feet
    % x1 = (3.28)^2*x2 --> if you can show a linear connection between two variables the matrix is non-invertible

- Too many features (e.g. m <= n)
  + Delete some features, or use regularization

-- Working on and Submitting Programming Assignments

>> submit()
Enter your choise [1-9]: 1
Login (Email address):
Password:

QUIZ

for i = 1:7
  for j = 1:7
    A(i, j) = log(X(i, j));
    B(i, j) = X(i, j) ^ 2;
    C(i, j) = X(i, j) + 1;
    D(i, j) = X(i, j) / 4;
  end
end

--> looping throughs rows and columns of a matrix

-- Week 3

== Classification and Representation ==

[Classification]

- Email: Spam/Not-Spam
- Online Transactions: Fraudulent (Yes/No)?
- Tumor: Malignant/Benign

y = {1, 0} % binary classification

0: 'Negative Class' % 0 is abscence
1: 'Positive Class' % 1 is presence

multi-class classification: y = {0, 1, 2, 3}

hθ(x) = θ^T*x

Threshold classifer output hθ(x) at 0.5:
- If hθ(x) >= 0.5, predict 'y=1'
- If hθ(x)  < 0.5, predict 'y=0'

Classification y = 0 or 1 % whereas:
Normal Regression: hθ(x) can be > 1 or < 0 % however:
Logistic Regression: 0 <= hθ(x) <= 1 % output logistic regression always between 0 and 1 | it is a classification algorithm

[Logistic Regression: Hypothesis Representation]

hθ(x) = θ^T*x % normal regression
hθ(x) = g(θ^T*x) % logistic regression
g(z) = 1 / 1 + e^-z % Sigmoid/Logistic Function --> z = θ^T*x
hθ(x) = 1 / (1 + e^(-θ^T*x)) % rewritten logistic regression function

Interpretation of Hypothesis Output

- hθ(x) = estimated probability that y = 1 on input x.
- Example:

If x = [ x0 ] = [     1     ]
       [ x1 ]   [ tumorSize ]

hθ(x) = 0.7

Tell patient that 70 percent chance of tumor being malignant (y = 1).

hθ(x) = P(y=1|x;θ) % probability of y=1, given x is parameterized by θ

P(y=0|x;θ) + P(y=1|x;θ) = 1
P(y=0|x;θ) = 1 - P(y=1|x;θ)

[Decision Boundary]

hθ(x) = g(θ^T*x) % outputs P(y=1|x;θ)
g(z) = 1 / (1+e^-z) % z = θ^T*x

Suppose predict 'y = 1' if hθ(x) >= 0.5
        predict 'y = 0' if hθ(x) < 0.5

g(z) >= 0.5 when z >= 0
hθ(x) = g(θ^T*x) >= 0.5 whenever θ^T*x >= 0 % θ^T*x = z

vice-versa

g(z) < 0.5 when z < 0
hθ(x) = g(θ^T*x) whenever θ^T*x < 0 % θ^T*x = z

hθ(x) = g(θ0 + θ1x1 + θ2x2)

    [ -3 ]
θ = [  1 ]
    [  1 ]

Predict 'y = 1' if -3 + x1 + x2 >= 0 % θ0 = -3, θ1 = 1, θ2 = 1, hence this equals θ^T*x
This is the same as: x1 + x2 >= 3 % you can plot a straight line where: x1 + x2 = 3 --> this line is called the "Decision Boundary"

Non-Linear Decision Boundaries

hθ(x) = g(θ0 + θ1x1 + θ2x2 + θ3x1^2 + θ4x2^2)

    [ -1 ]
    [  0 ]
θ = [  0 ]
    [  1 ]
    [  1 ]

Predict 'y = 1' if -1 + x1^2 + x2^2 >= 0 % x1^2 + x2^2 >= 1 --> x1^2 + x2^2 = 1 | Decision boundary (circle shaped)

- The Decision Boundary is a property of the parameters theta, NOT the underlying data.
- When you take even higher order polynomials, e.g. θ5x1^3*x2, you can get very complex/unique decision boundaries.

== Logistic Regression Model ==

[Cost Function]

Training Set: {x^(1), y^(1), x^(2), y^(2), ..., x^(m), y^(m)}

                 [ x0  ]
m examples: x  = [ x1  ] % n + 1 dimensional vector
                 [ ... ]
                 [ xn  ]

x0 = 1, y = {1, 0}

hθ(x) = 1 / (1 + e^-(θ^T*x))

How to choose parameters θ?

Cost Function

Linear Regression: J(θ) = 1 / m * Σ 1/2 * (hθ(x^(i)) - y^(i))^2 % rewrite 1/2 * (hθ(x^(i)) - y^(i))^2 as cost(hθ(x), y)

Logistic Regression: Cost(hθ(x), y) = { -log(hθ(x))     if y = 1
                                      { -log(1 - hθ(x)) if y = 0
- Cost = 0 if y = 1 & hθ(x) = 1
- But as hθ(x) moves to 0, the cost moves to infinity
- This captures the intuition that if hθ(x) = 0 (predict P(y=1|x;θ) = 0), buy y = 1, we will
  penalize the learning algorithm by a very large cost.

[Simplified Cost Function and Gradient Descent]

Logistic Regression Cost Function

J(θ) = 1 / m * Σ Cost(hθ(x^(i)), y^(i))

Cost(hθ(x), y) = = { -log(hθ(x))     if y = 1
                   { -log(1 - hθ(x)) if y = 0

Note: y = 0 or 1 always

Cost(hθ(x), y) = -y*log(hθ(x)) - (1-y)*log(1-hθ(x))

If y = 1: Cost(hθ(x), y) = -1*log(hθ(x)) - (1-1)log(1-hθ(x)) % second part becomes 0, cancels out
        : -log(hθ(x))
If y = 0: Cost(hθ(x), y) = -0*log(hθ(x)) - (1-0)log(1-hθ(x))
        : -log(1-hθ(x))

J(θ) = 1/m * Σ Cost(hθ(x^(i)), y^(i))
==
J(θ) = -1/m * [Σ y^(i)log(hθ(x^(i))) + (1-y^(i))log(1-hθ(x^(i)))]

To fit parameters θ:

min J(θ)
 θ

To make a prediction given new x:

Output hθ(x) = 1 / (1 + e^-(θ^T*x)) % P(y=1|x;θ)

Gradient Descent

% Follows the following procedure:

repeat {
  θj := θj - α*((∂/∂θj) * J(θ))
}

% Partial derivative calculation for paramater j

∂/∂θj J(θ) = 1/m * Σ(hθ(x^(i))-y^(i)) * xj^(i)

% Plug partial derivative it the Gradient Descent formula

repeat {
  θj := θj - α * 1/m * Σ(hθ(x^(i))-y^(i)) * xj^(i) % θj := θj - α/m * Σ(hθ(x^(i))-y^(i)) * xj^(i)
}

[Advanced Optimization]

'Conjugate gradient', 'BFGS', and 'L-BFGS' are more sophisticated, faster ways to optimize theta instead of using gradient descent. A. Ng suggests you do not write these more sophisticated algorithms yourself (unless you are an expert in numerical computing) but use them pre-written from libraries. Octave provides them.
We first need to provide a function that computes the following two equations:

J(θ)
∂/∂θj J(θ)

We can write a single function that returns both of these:

function [jVal, gradient] = costFunction(theta)
  jval = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end

Then we can use octave's 'fminunc()' optimization algorithm along with the 'optimset()' function that creates an object containing the options we want to send to 'fminunc()'.

options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);

We give to the function 'fminunc()' our cost function, our initial vector of theta values, and the 'options' object that we created beforehand.

== Multiclass Classification ==

[Multiclass Classification: One-vs-All/One-vs-Rest]

hθ^(i)(x) = P(y=i:x;θ) (i = 1, 2, 3)

hθ^(1)(x) --> learns probability y = 1
hθ^(2)(x) --> learns probability y = 2
hθ^(3)(x) --> learns probability y = 3

- We get three classifiers that learn to recognize 1, 2, 3 respectively.
- You train a logistic regression classifer hθ^(i)(x) for each class i to predict the probability that y = i.
- On a new input x, to make a prediction, pick the class i that maximizes max hθ^(i)(x) --> get the class for which the probability is highest.

-- Week 4

== Neural Networks: Representation ==

[Motivations: Non-Linear Hypotheses]

[Motivations: Neurons and the Brain]

- Origins: Algorithms that try to mimic the brain.
- Was very widely used in the 80's and early 90's; popularity diminished in late 90's.
- Recent resurgence: State-of-the-art technique for many applications.

[Neural Networks: Model Representation I]

Neuron Model: Logistic unit
- Sigmoid (logistic) activation function % g(z)
- Weights == parameters

Neural Network
- Layer 1: Input Layer
- Layer 2: Hidden Layer % it's not X, not y hence we don't see them; hidden
- Layer 3: Output Layer

ai^(j) = 'activation' of unit i in layer j
θ^(j) = matrix of weights controlling function mapping from layer j to layer j + 1

[Neural Networks: Model Representation II]

[Applications: Examples and Intuitions I]

y = x1 XOR x2 --> true if x1 or x2 is equal to 1, but not both (XOR = exclusive OR)
    x1 XNOR x2 --> true of both are 1 or both are 0 % NOT (x1 XOR x2)

Simple example: AND
x1, x2 C- {0, 1}
y = x1 AND x2

[Applications: Examples and Intuitions II]

Negating: put large negative values before x
- (NOT x1) AND (NOT x2) % put large negative integers before them
  + This only defaults to true when x1 = x2 = 0

-- Week 5

== Neural Networks: Learning ==

[Cost Function and Backpropagation]

[Unrolling Parameters]

s1 = 10
s2 = 10
s3 = 1

thetaVec = [ Theta1(:); Theta2(:); Theta3(:)]; % unroll items in one long vector
DVec = [ D1(:); D2(:); D3(:) ];
Theta1 = reshape(thetaVec(1:110), 10, 11); % reshape first 110 items into a 10x11 matrix
Theta2 = reshape(thetaVec(111:220), 10, 11);
Theta3 = reshape(thetaVec(221:231), 1, 11);

- Have initial parameters Theta1, Theta2, Theta3.
- Unroll to get initialTheta to pass to 'fminunc(@costFunction, initialTheta, options)'

function [jval, gradientVec] = costFunction(thetaVec)

- From thetaVec, get Theta1, Theta2, Theta3 (using reshape functions).
- Use the matrices to forward prop/back prop to compute D1, D2, D3 and J(Theta) (jval).
- Unroll D1, D2, D3 to get gradientVec (which the function also returns).

[Gradient Checking]

gradApprox = (J(theta + EPSILON) - J(theta - EPSILON)) / (2 * EPSILON) % this should be aproximately equal to your gradient
% epsilon = 10^-4

Parameter vector θ

θ C- R^n (E.g. θ is 'unrolled' version of Theta1, Theta2, Theta3)
θ = [θ1, θ2, θ3, ..., θn]

∂/∂θ1 J(θ) = (J(θ1 + EPSILON, θ2, θ3, ..., θn) - J(θ1 - EPSILON, θ2, θ3, ... θn)) / (2*EPSILON % etc.

for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) = thetaPlus(i) + EPSILON;
  thetaMinus = theta;
  thetaMinus(i) = thetaMinus(i) - EPSILON;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus)) / (2 * EPSILON);
end;

Check that gradApprox = DVec (our gradients unrolled into a vector; which we got from backpropagation)

Implementation Note:
- Implement backprop to compute DVec (unrolled D^(1), D^(2), D^(3))
- Implement numerical gradient check to compute gradApprox
- Make sure they give similar values
- Turn off gradient checking. Using backprop code for learning

Important:
- Be sure to disable your gradient checking code before training your classifier.
  If you run numerical gradient computation on every iteration of gradient descent (or in
  the inner loop of costFunction()) your code will be very slow

[Random Initialisation]

Initial value of Theta

- For gradient descent and advanced optimisation method, need initial value for Theta

optTheta = fminunc(@costFunction, initialTheta, options)

Random initialization: Symmetry breaking

Initialize each value of Theta to a random value in [-EPSILON, EPSILON]

E.g.:

Theta1 = rand(10,11) * (2*INIT_EPSILON) - INIT_EPSILON % random 10 * 11 dimensional matrix (values between 0 and 1)
Theta1 = rand(11,11) * (2*INIT_EPSILON) - INIT_EPSILON % the latter part causes it to be a value within [-EPSILON, EPSILON]

[Putting it Together]

Training a neural Network
- Number of input units: Dimension of features x^(i)
- Number of output units: Number of classes
- Reasonable default: 1 hidden layer, or if > 1 hidden layer, have same number of hidden units
  in every layer (usually the more the better)

1 - Randomly initialize weights
2 - Implement forward propagation to get hθ(x^(i)) for any x^(i)
3 - Implement code to compute cost function J(θ)
4 - Implement backprop to compute partial derivatives ∂/∂Thetajk^(l) J(Theta)

for i = 1:m

  Perform forward propagation and backpropagation using exmaple(x^(i), y^(i))
  Get activations a^(l) and delta terms ∂^(l) for l = 2, ..., L

5 - Use gradient checking to compare ∂/∂Thetajk^(l) J(Theta) computed using backpropagation vs. using numerical estimate of gradient of J(Theta).
    Then disable gradient checking code.
6 - Use gradient descent or advanced optimization method with backpropogation to try to minimize J(Theta) as a functions of parameters Theta.

-- Week 6

== Advice for Applying Machine Learning ==

[Evaluating a Learning Algorithm]

Deciding What to Try Next

Debugging a learning algorithm:
- Suppose you have implemented regularized linear regression to predict housing prices.
- However, when you test your hypothesis on a new set of houses, you find that it makes
  unacceptably large errors in its predictions. What should you try next?
  + Get more training examples
  + Try smaller sets of features
  + Try getting additional features
  + Try adding polynomial features
  + Try decreasing/increasing lambda (regularization)

Machine learning diagnostic
- Diagnostic: a test that you can run to gain insight what is/is not working with a learning
  algorithm, and gain guidance as to how best to improve its performance.
- Diagnostics can take time to implement, but doing so can be a very good use of your time.

Evaluating a Hypothesis

- Training Set: random 70%
- Test Set (Mtest / Xtest / Ytest): random 30%

Training/testing procedure for linear regression
- Learn parameter θ from training data (minimizing training error J(θ)) % with 70% of the data
- Compute test set error (Jtest(θ)): the average squared error of your test set

Training/testing procedure for logistic regression
- Learn parameter θ from training data
- Compute test set error
- Misclassification error (0/1 misclassification error)
  + err(hθ(x),y) {1 if hθ(x >= 0.5 & y = 0  OR if hθ(x) < 0.5 & y = 1) ELSE 0} % the fraction of all mtest that were mislabeled

Model Selection and Train/Validation/Test Sets

Once parameters θ0, θ1, ..., θ4 were fit to some set of data (training set),
the error of the parameters as measured on that data (the training error J(θ))
is likely to be lower than the actual generalization error.

Model Selection
- d = degree of polynomial
- Use various degrees of polynomials
  + Look at respective Jtest(θ) values --> which model has the lowest error?
  + Problem: optimal Jtest(θ) is likely to be an optimistic estimate of generalizaiton error.
    I.e. our extra parameter (d = degree of polynomial) is fit to test set.
    % Say we look at 10 model versions and select only the one that fits best on the test set --> biased to the test set!
  + To address this problem, we split the dataset in 3 pieces:
    * Training Set % 60%
    * Cross Validation (CV) Set % 20%
    * Test Set % 20%
  + Train/Validation/Test Error --> All use the same J formula
    i. You train on the Training Set
    ii. You pick the model (parameter vector θ) with the lowest CV error
    iii. Estimate generalization error of the model using the test set

[Bias vs. Variance]

Diagnosing Bias (Underfitting) vs. Variance (Overfitting)

Suppose your learning algorithm is performing less well than you were hoping.
Either Jcv(θ) or Jtest(θ) is high. Is it a bias problem or a variance problem?

Bias (underfit):
- Jtrain(θ) will be high
- Jcv(θ) will be high % Jcv(θ) will approximately be equal to Jtrain(θ)

Variance (overfit):
- Jtrain(θ) will be low
- Jcv(θ) will be high (again) % Jcv(θ) >> Jtrain(θ)

Regularization and Bias/Variance

- Train multiple models with various values of Lambda % e.g. 0, 0.01, 0.02, 0.04, up to 10.24
- Look at Jcv(θ), pick lowest one
- Then use it on Jtest(θ)

- Small lambda: risk of overfitting (high variance)
- Large lambda: risk of underfitting (high bias)

Learning Curves

- You deliberately limit your training set size
- When m is low Jtrain(θ) is going to be 0 (it is easy to perfectly fit a line)
  + The larger m, the larger the average error of Jtrain(θ)
- In contrast, the average error of Jcv(θ) will decrease with m

High Bias
- When fitting a straight line, getting more data will not help
  + Only at the beginning you will slightly reduce Jcv(θ) with more m
  + Jtrain(θ) will apprach Jcv(θ) with more m
  + Both train/cv average error values are high
  + Both slopes flatten out, so more data has no effect
- If a learning algorithm is suffering from high bias, getting more
  training data will not (by itself) help much

High variance
- Jtrain(θ) will slightly increase with m % it becomes harder to fit the data well
- Jcv(θ) will slightly decrease with m % so the curves are converging --> more data will help reduce the gap
  + There will be a gap between Jcv(θ) & Jtrain(θ)
- If a learning algorithm is suffering from high variance, getting more data
  is likely to help

Deciding What to Do Next Revisited

Debugging a learning algorithm:
Suppose you have implemented regularized linear regression to predict housing prices.
However, when you test your hypothesis in a new set of houses, you find that it makes
unacceptably large errors in its prediction. What should you try next?

  + Get more training examples --> fixes high variance % when high bias this is pointless
  + Try smaller sets of features --> fixes high variance % when high bias this is pointless
  + Try getting additional features --> fixes high bias % current hypothesis is too simple, so get additional features to better fit the training set
  + Try adding polynomial features --> fixes high bias
  + Try decreasing lambda --> fixes high bias
  + Try increasing lambda (regularization) --> fixes high variance

Neural networks and overfitting

- 'Small' neural networks: fewer parameters; more prone to underfitting % computionally cheaper
- 'Large' neural networks: more parameters (hidden units/layers); more prone to overfitting % computationally more expensive

Use regularization to address overfitting.

== Machine Learning System Design ==

[Building a Spam Classifier]

Prioritizing What to Work On

Building a spam classifier
- Supervised learning
- x = features of email
- y = spam (1) or not spam (0) % ham/spam
- Features x: choose 100 words indicative of spam/not spam
  + E.g. deal, buy, discount --> spam
  + E.g. andrew, now, ... --> not-spam
  + Take the list of 100 words, sort in alphabetical order
    * When appears: 1, if not-appears 0
    * Note: In practice, take most frequently occurring n words (10000 to 50000)
      in training set rather than manually pick 100 words
- How to spend your time to make it have low error?
  + Collect lots of data
    * E.g. 'honeypot' project
  + Develop sophisticated features based on email routing information (from email header)
  + Develop sophisticated features for message body, e.g. should 'discount' and 'discounts'
    be treated as the same word? How about 'deal' and 'dealer' ? Features about punctuation?
  + Develop sophisticated algorithm to detect misspellings (e.g. m0rtgage, med1cine, w4tches)

Error Analysis

Recommended Approach:
- Start with a simple algorithm that you can implement quickly.
  Implement it and test it on your cross-validation data.
- Plot learning curves to decide if more data, more features, etc.
  are likely to help.
- Error analysis: manually examine the examples (in the cross validation set)
  that your algorithm made errors on. See if you spot any systematic trend in
  what type of examples it is making errors on. % this is the process where you get inspiration to create new features

Example:
- mcv = 500 examples in cross validation set
- Algorithm misclassifies 100 emails
- Manually examine the 100 errors, and categorize them based on:
  + What type of email it is
    * E.g. Pharma, Replica/Fake, Steal passwords, etc. % highest frequency category: focus attention there
  + What cues (features) you think would have helped the algorithm
    classify them correctly

The importance of numerical evaluation % makes process easier
- Should discount/discounts/discounted/discounting be treated as the same word?
- Can use 'stemming' software (e.g. 'Porter stemmer') % to let you treat all of the above words as the same words; basically only looking at the first few characters
  + Can make mistakes: universe/university
- Error analysis may not be helpful for deciding if this is likely to improve performance.
  Only solution is to try it and see if it works.
- Need numerical evaluation (e.g., cross validation error) of algorithm's performance
  with and without stemming:
  + With stemming: 3% classification error
  + Without stemming: 5% classification error
    * Easy to grasp that stemming improves performance
- Second example: distinguish upper vs. lower case (Mom/mom)

[Handling Skewed Data]

Error Metrics for Skewed Classes % way more of either label 1 or 0

Cancer classification example
- Train logistic regression model hθ(x).
  + y = 1 if cancer, y = 0 otherwise
- Find that you got 1% error on test set
  + 99% correct diagnoses
- Only 0.50% of patients have cancer --> skewed classes, we have many more instances of one class (no-cancer)

function y = predictCancer(x)
  y = 0; % ignore x
return

- This non-learning function would return a 0.5% error...!
- Accuracy is not a good evaluation metric in this case.

Precision/Recall
- y = 1 in presence of rare class that we want to detect
- Actual vs. predicted outcome:
  + True Positive % TP
  + True Negative % TN
  + False Positive % FP / Type I Error
  + False Negative % FN / Type II Error
- Precision: Of all patients where we predicted y = 1, what fraction actually has cancer?
  + True Positives / # of Predicted Positives == TP / (TP + FP) % predictions where you were both right and wrong == predicted positive
- Recall: Of all patients that actually have cancer, what fraction did we correctly detect as having cancer?
  + True Positives / # of Actual Positives == TP / (TP + FN)

Trading Off Precision and Recall % adjusting thresholds

- Logistic regression: 0 <= hθ(x) <= 1
  + Predict 1 if hθ(x) >= 0.5
  + Predict 0 if hθ(x) < 0.5
- Suppose we want to predict y = 1 (cancer) only if very confident. We can adjust the thresholds:
  + Predict 1 if hθ(x) >= 0.7 % or even 0.9
  + Predict 0 if hθ(x) < 0.7 % or even 0.9
    * Higher precision, but lower recall
- Suppose we want to avoid missing too many cases of cancer (avoid false negatives)
  + Predict 1 if hθ(x) >= 0.3 % or even 0.1
  + Predict 0 if hθ(x) < 0.3 % or even 0.1
    * Lower precision, but higher recall

Generally: predict 1 if hθ(x) >= threshold.

Can we choose the threshold value automatically?

F1 Score (F Score) % take the P & R values of the CV set
- How to compare precision/recall numbers?
  + We now no longer have 1-value evaluation, since we have 2 numbers.
- We could use:
  + The average: P + R / 2 % not great, when y equals 1 all the time the average can still be around 0.50
  + F1 Score: (2 * P * R) / (P + R) % it combines P & R, but moves away from 0 values for either values (since it takes the product)
    * If P = 0 OR R = 0 --> F-score = 0
    * If P = 1 AND R = 1 --> F-score = 1
- Ultimately you could use a range of P & R values, and then automatically take the respective
  P & R values of the highest F-score

[Using Large Data Sets]

Data For Machine Learning

Designing a high accuracy learning system
- E.g. classify between confusable words
  + E.g. {to, two, too}, {then, than}
- For breakfast I ate ____ eggs.
- Algorithms:
  + Perceptron (Logistic regression)
  + Winnow
  + Memory-based
  + Naïve Bayes

'It is not who has the best algorithm that wins.
                               it is who has the most data.'

Large data rationale
- Assume feature x C- R^n+1 has sufficient information to predict y accurately.
  + Example: For breakfast I ate ____ eggs. % human can predict
  + Counterexample: Predict housing price from only size (feet^2) and no other features. % human cannot determine
- Useful test: Given the input x, can a human expert confidently predict y?
- Use a learning algorithm with many parameters (e.g. logistic regression/linear regression
  with many features; neural network with many hidden units). % low bias algorithms, fit complex functions
  * Chances are Jtrain(θ) will be small % training error small
- So when we use a very large training set, we are still unlikely to overfit
  * So Jtrain(θ) is roughly equal to Jtest(θ)
  * And so Jtest(θ) will also be small (given Jtrain(θ) should be small, and they are roughly equal)
    - We want low bias and low variance
      + We address bias (underfit) by taking many parameters/hidden units (complex function)
      + We address variance (overfit) by having many data

-- Week 7

== Support Vector Machines ==

% Large Margin Classification

[Optimization Objective]

Alternative view of logistic regression
- If y = 1, we want hθ(x) = 1, θ^Tx >> 0
- If y = 0, we want hθ(x) = 0, θ^Tx << 0

- cost1(z) % cost when y = 1
- cost0(z) % cost when y = 0
  + z = θ^Tx

- We drop 1/m & lambda/2m --> lambda/2 remains. % This does not alter anything, we only did this before for the sake of the partial derivates.

Logistic Regression: A + Lambda * B % Lambda contains your parameters
  + By changing lambda you can trade-off how much you want to fit the training set well (by minimizing A)
    versus how much we care about keep the values of the parameters small (& fitting new unseen data better)
  + Setting a large value of Lambda gives B a very high weight

SVM: C * A + B % Lambda turns into C & order changes
  + Called C by convention
  + If we set C to a very small value, that corresponds to giving B a very large weight (larger than A)
  + Hence we have a different method to alter the weights of A & B
  + C = 1 / Lambda

min C * Σ[y^(i)*cost1(θ^Tx^(i)) + (1-y^(i))cost0(θ^Tx^(i))] + 1/2 * Σ θ^2
 θ

Hypothesis: hθ(x) {1 % if θ^Tx >= 0
                  or
                  0} % if θ^Tx < 0 | 0 otherwise
                  % it does not output a probability like logistic regression, just binary 1 or 0

[Support Vector Machines] % Large margin classifier, it tries to separate the data with a large margin

- If y = 1, we want θ^Tx >= 1 (not just >= 0) % since from 1 onwards z = 0
- If y = 0, we want θ^Tx <= -1 (not just < 0) % since from -1 onwards z = 0

% Do not just barely get the hypothesis right (θ^Tx >= 1 or θ^Tx <= -1)
% The 1/-1 in this case is the 'safety margin' it uses (of range 2)

- A SVM tries to separate the negative and positive examples with a margin
  that is as large as possible. % This somehwat means the decision boundary as as optimal as possible.

When C is very large (e.g. 100,000)
- Whenever y^(i) = 1: θ^Tx >= 1
- Whenever y^(i) = 0: θ^Tx <= -1
- min C * 0 + 1/2 * Σ θ^2
  + Subject to:
    * θ^Tx >= 1 if y^(i) = 1
    * θ^Tx <= -1 if y^(i) = 0

SVM Decision Boundary: Linearly Seperable Case/Example
- It will create a decision boundary (separation line) with the largest margin/distance between groups

Large Margin Classifier in Presence of Outliers
- If C is very large, it will optimise the decision boundary to fit all relevant values (overfitting) --> it needs A to be small
  + In the presence of outliers, this will make it prone to fit them all in (overfit)
- If C is small, it will not do this and will stick to a more robust/natural boundary (will perform better on new data)

[Mathematics Behind Large Margin Classification]

-- Vector Inner Product --

u = [u1 ; u2]
v = [v1 ; v2]
u^T*v = Vector Inner Product
||u|| = lenght of vector u = (u1^2+u2^2)^0.5
p = length of projection of v onto u % p is signed, it can be positive or negative
% if the angle between v and u is greater than 90 degrees, p is negative
u^T*v = p * ||u|| == u1*v1 + u2*v2 == v^T*u % all the same

-- SVM Decision Boundary --

% θ^Tx^(i) = p^(i) * ||θ|| = θ1*x1^(i) + θ2*x2^(i)

min Σ θ^2
 θ

- Subject to:
  + θ^Tx >= 1 if y^(i) = 1
  + θ^Tx <= -1 if y^(i) = 0
- Simplification:
  + θ0 = 0
  + n = 2

Given our example, we can now write that

                                            ----- ||θ|| -----
min Σ θ^2 == 1/2 * (θ1^2 + θ2^2) ==  1/2 * ((θ1^2 + θ2^2)^0.5)^2 == 1/2 * ||θ||^2
 θ

θ^Tx == u^Tv
θ^Tx^(i) == p^(i) * ||θ|| == θ1 * x1 + θ2 * x2
- p^(i) * ||θ|| >=  1 if y^(i) = 1
- p^(i) * ||θ|| <= -1 if y^(i) = 0

- Where p^(i) is the projection of x^(i) onto the vector θ (which is a 90 degree line on the decision boundary).
- You want p to be large so you can set small values of theta (which you try to minimize).
- This is where the margin comes from!

% The simplification of θ0 = 0 only causes all decision boundaries to pass through the 0,0 origin.

% Kernels

[Kernels I]

Non-linear Decision Boundary

In the case of a non-linear decision boundary, a way to allow for this is using a complex set of polynomials:

Predict y = 1 if
θ0 + θ1x1 + θ2x2 + θ3x1x2 + θ4x1^2 + θ5x2^2 + ... >= 0 % etc.
      f1     f2      f3       f4       f5              % etc.
      x1     x2     x1x2     x1^2      x2^2            % etc.

hθ(x) = {1 if θ0 + θ1x + ... >= 0
         0 if θ0 + θ1x + ... < 0 | otherwise}

Is there a different/better choice of the features f1, f2, f3, ... ?

Given x, compute new feature depending on proximity to landmarks l^(1), l^(2), l^(3)

Given x:
f1 = similarity(x, l^(1)) = exp(- ((||x - l^(1)||^2)/2σ^2)) % exp() = e^
f2 = similarity(x, l^(2)) = exp(- ((||x - l^(2)||^2)/2σ^2)) % ||x - l^(i)||^2 == euclidian distance^2 between points x & l
f3 = similarity(x, l^(3)) = exp(- ((||x - l^(3)||^2)/2σ^2))
... etc.

- Similarity function --> Kernel % our example is a Gaussian Kernel (using sigma/stdev)
- Notation: k(x, l^(i))

Kernels and Similarity

f1 = similarity(x, l^(1)) = exp(- (||x - l^(1)||^2)/2σ^2) = exp((Σ(xj-lj^(1)))/2σ^2)

If x ≈ l^(1) : % is x is close to l^(1)
                f1 ≈ exp(- 0^2 / 2σ^2) ≈ e^0 / e^-0 ≈ 1 % since then x - l^(1) will be close to 0; when x & l are close to one another

If x is far from l^(1):
                f1 ≈ exp(-((large number)^2/2σ^2)) ≈ 0

-- Example --

l^(1) = [ 3 ], f1 = exp(- ((||x - l^(1)||^2)/2σ^2))
        [ 5 ]

% Let's use: σ^2 = 1

- Every feature is measuring how close x is to its landmark
- The smaller σ^2 (variance), the smaller the range of x-values that will default to 1 for this specific feature, conversely a higher σ^2 will classify a larger range of x-values as close to the feature (classify as 1)

- Predict 1 when: θ0 + θ1f1 + θ2f2 + θ3f3 >= 0
- Let us assume that: θ0 = -0.5, θ1 = 1, θ2 = 1, θ3 = 0 % so θ1 & θ2 predict positive, θ3 predicts negative
  + Training values close to l^(1) & l^(2) (θ1 & θ2) will therefore be predicted as positive, others negative
- Since a training example x is close to l^(1) (f1), we get: -0.5 + 1*1 + 1*0 + 1*0 = 0.5 >= 0 % so predict positive: y = 1

Questions:
- How do we choose our landmark positions?
- What other similarity functions can be used other than the one we used (the Gaussian Kernel)?

[Kernels II]

- We put landmarks on exactly all the locations of our training examples (both positive and negative)
  + So we get l^(m) landmarks, 1 per location for each of the training examples
  + Hence, my features will measure how close an example is to something I had in my training set

SMV with Kernels
Given (x^(1), y^(1)), (x^(2), y^(2)), ... , (x^(m), y^(m)),
choose l^(1) = x^(1), l^(2) = x^(2), ... , l^(m) = x^(m)

Given example x:
f1 = similarity(x, l^(1))
f2 = similarity(x, l^(2))
...

     [ f0 ]
     [ f1 ]
f =  [ f2 ]
     [ .. ]
     [ fm ]

For training example (x^(i), y^(i)):
f1^(i) = sim(x^(i), l^(1))
f2^(i) = sim(x^(i), l^(2))
... % --> somehwere you will have: fi^(i) = sim(x^(i), l^(i)) --> basically checking distance from itself, which will be 1 and is fine
fm^(i) = sim(x^(i), l^(m))

SVM with Kernels

Hypothesis: Given x, compute features f C- R^m+1
- Predict 'y=1' if  θ^T * f >= 0 % θ0f0 + θ1f1 + ... + θmfm --> θ C- R^m+1

                                                                    n % in this case n = m
min C * Σ[y^(i)*cost1(θ^Tf^(i)) + (1-y^(i))cost0(θ^Tf^(i))] + 1/2 * Σ θ^2 % the last term can be written as θ^T*θ if we ignore θ0
 θ                                                                 j = 1 % we do not regularize θ0

- As a result, it will give theta weights of 0 for landmarks that are negative, and 1 that are positive?
- In this case, the number of features is equal to the number of training examples

SVM parameters:

C %= 1 / lambda)
- Large C: Lower bias, high variance % prone to overfit
- Small C: Higher bias, low variance % prone to underfit

σ^2
- Large σ^2: features fi vary more smoothly.
  + Higher bias, lower variance
- Small σ^2: features fi vary less smoothly.
  + Lower bias, higher variance

% SVMs in Practice

[Using an SVM]

Use SVM software package (e.g. liblinear, libsvm, ...) to solve for parameters θ

Need to specify:
- Choice of parameter C
- Choice of kernel (similarity function)

Linear Kernel % linear kernel
- Predict 'y = 1' if θ^Tx > 0 % θ0 + θ1x1 + ... + θnxn >= 0
- Version of the SVM that just gives you a standard linear classifier with a linear decision boundary
- Could be useful when n is large but m is small (not enough data)

Gaussian Kernel
- fi = exp(-((||x - l^(i)||^2)/2σ^2)), where l^(i) = x^(i)
- Need to choose σ^2
  + Large: high bias, low variance
  + Low: low bias, large variance

Kernel (similarity) functions:

function f = kernel(x1, x2)

f = exp(- ((||x1 - x2||^2) / 2σ^2))
% x1 = training example
% x2 = landmark

Note: Do perform feature scaling before using the Gaussian Kernel.
- Imagine computing ||x - l||^2
- v = x - l
- ||v||^2 = v1^2 + v2^2 + ... + vn^2 == (x1 - l1)^2 + (x2 - l2)^2 + ... + (xn - ln)^2
                                         1000 feet^2   1-5 bedrooms
- If you do not feature scale first, the larger range of the size of the house in the feature house_size,
  will dominate other features such as the number of bedrooms

Other choices of Kernel
- Note: not all similarity functions similarity(x, l) make valid kernels.
  They need to satisfy a technical condition called 'Mercers Theorem' to make sure
  SVM packages' optimizations run correctly, and do not diverge.

Many off-the-shelf kernels available:
- Polynomial kernel: k(x, l) = (x^T*l)^2 % if x and l tend to be close to each other, the inner product tends to be large
                     (x^T * l + 1)^3 % (x^T * l + constant)^degree
  + Can be used with strictly non-negative numbers (so no issues with inner product)
  + Captures the intuition that when x and l tend to be close to one another, to inner product tends to be large
- More esoteric: String kernel, chi-square kernel, histogram intersection kernel

Multi-class classification
- Many SVM packages already have built-in multi-class classification functionality
- Otherwise, use one-vs.-all method.
  + Train K SVMs, one to distinguish y = i from the rest, for i = 1, 2, ..., K) % y C- {1, 2, 3, ..., K}
  + Get θ^(1), θ^(2), ..., θ^(K).
  % θ^(1) will predict the chance that y = 1
  % θ^(2) will predict the change that y = 2, etc.
  + Pick class i with largest (θ^(i)^T*x)

Logistic regression vs. SVMs
- n = number of features (x C- R^n+1)
- m = number of training examples

If n is large (relative to m):
- Use logistic regression, or SVM without a kernel ('linear kernel')
  % E.g. n >= m, n = 10.000, m = 10 to 1000
  % For instance with a spam classifier, where you have 10.000 words and just 1000 messages
  % You do not have enough data to fit a non-linear model
- If n is small, m is intermediate
  + Use SVM with Gaussian kernel
  % n = 1-1000, m = 10-10.000/50.000
- If n is small, m is large
  + Create/add more features, then use logistic regression or SVM without a kernel
  % n = 1-1000, m = 50.000+
  % Logistic regression looks a lot like a SVM without a kernel
- Neural network likely to work well for most of these settings, but may be slower to train.

-- Week 8

== Unsupervised Learning ==

% Clustering

[Unsupervised Learning: Introduction]

Applications of clustering:
- Market segmentation
- Social network analysis
- Organise computing Clusters
- Astronomical data analysis

[K-Means Algorithm]

- Random initialize a certain number of centroids (how many clusters you want)
- On every iteration it assigns the training examples to a single centroid closest to it
- It then calculates the mean of all examples per centroid group, and moves the centroid to that location
- Next iteration, etc.

Input:
- K (number of clusters) % we will see later on how to determine K
- Training set {x^(1), x^(2), ..., x^(m)} % no labels anymore

x^(i) C- R^n % (drop x0 = 1 convention)

Randomly initialise K cluster centroids μ1, μ2, ..., μK C- R^n

Repeat {
        for i = 1 to m
          c^(i) := index(from 1 to K) of cluster centroid closest to x^(i) % Cluster assignment step (which centroid it is closest to)
          --> min ||x^(i) - μk||^2 to calculate distance between x^(i) and μk lower case k % normal or squared distance (^2) is by convention ^2, but should yield same result
               k                 the value of k that minimizes the distance will be chosen as c^(i)
        for k = 1 to K
          μK := average (mean) of points assigned to cluster k % Move centroid step
          - c^(1) = 2, c^(5) = 2, c^(6) = 2, c^(10) = 2
          - μ2 = 1/4 * [x^(1) = 2, x^(5) = 2, x^(6) = 2, x^(10)] C- R^n
          - What if there is a cluster without points? You eliminate that cluster, so you now have K-1 clusters.
            + The other thing you can do is randomly re-initialise it (less common).
       }

[Optimization Objective]

K-means optimization objective

- c^(i) = index of clusters(1, 2, ..., K) to which example x^(i) is currently assigned
- μK = cluster centroid k (μK C- R^n)
- μc^(i) = cluster centroid of cluster to which example x^(i) has been assigned

J(c^(1), ..., c^(m), μ1, ..., μK) = 1/m Σ ||x^(i) - μc^(i)||^2

- Distortion Cost Function:

        min         J(c^(1), ..., c^(m), μ1, ..., μk)
c^(1), ..., c^(m)
   μ1, ..., μk

Randomly initialise K cluster centroids μ1, μ2, ..., μK C- R^n

Repeat {
        for i = 1 to m % Cluster assignment setp, minimize J(...) w.r.t. c^(1), c^(2), ... , c^(n) while hodling μ1, ..., μk fixed
          c^(i) := index(from 1 to K) of cluster centroid closest to x^(i)

        for k = 1 to K % move centroid step minimize J(...) w.r.t. μ^1, ..., μ^k
          μK := average (mean) of points assigned to cluster k
       }

- First mimise J w.r.t c, next you minimise J() w.r.t μ

[Random Initialization] % To avoid local optima

- Should have K < m
- Randomly pick K training examples
- Set μ1, ..., μk equal to these K examples

Local optima

- Try multiple random initialisations:

for i = 1 to 100 { % you typically run K-means 50-1000 times

  Randomly initialize K-mean.
  Run K-means. Get c^(1), ..., c^(m), μ1, ..., μk
  Compute cost functions (distortion)
  J(c^(1), ..., c^(m), μ1, ..., μk)

  }

Then, pick clustering that gave lowest cost (distortion): J(c^(1), ..., c^(m), μ1, ..., μk)
- The larger K, the less important it is to random initiate a lot of times (higher chance the first initialisation is already a decent solution)

[Choosing the Number of Clusters]

- In general you look at visualizations, etc. to try to come up with a good number yourself
- Elbow method: plot cost function J (y-axis) over K (number of clusters)
  + Where the elbow is, the curve bends most, that's the optimum number
  + However, oftentimes no clear 'elbow' to be found, so then you still do not know what to do
- Sometimes, you're running K-means to get clusters to use for some later/downstream purpose.
  Evaluate K-means based on a metric for how well it performs for that later purpose.
  + Basically start with the question you're trying to answer.

== Dimensionality Reduction ==

% Motivation

[Motivation I: Data Compression]

- Say you have data in both cm and inches --> reduce data from 2-D to 1-D since they are the same thing!
- Any two variables that measure the same thing and show this, should be reduced into one
  + Reduce data from 2-D to 1-D
    x^1 C- R^2 (two dimensional vector: x1, x2) --> z^(1) C- R (one dimensional vector, no two coordinates anymore)

This saves data, and fastens your learning algorithms

Reduce data from 3-D to 2-D
- x^(i) C- R^3 % i.e. you have 3 data points in each x^(i)
- We need z1 & z2 to specify a point on our new plane (as in field)
  + z^(i) C- R^2 % now we only have 2-coordinates left

[Motivation II: Visualization]

- If you have 50 features, x^(i) is a 50-dimensional vector!

% Principal Component Analysis

[Principal Component Analysis Problem Formulation]

- Goal: Reduce from 2-dimension to 1-dimension: Find a direction (a vector u^(1) C- R^n)
  onto which to project the data so as to minimize the projection error.
  + It does not matter whether the direction u is negative or positive
- Reduce from n-dimension to k-dimension: Find k vectors u^(1), u^(2), ... u^(k)
  onto which to project the data, so as to minimize the projection error.

PCA is not linear regression
- Linear regression tries to minimize the vertical squared distance (y) from the dots to the regression line
- PCA tries to minimize the projected squared distance of the dots to the line

[Principal Component Analysis Algorithm]

Data preprocessing

Training set: x^(1), x^(2), ..., x^(m)

Preprocessing (feature scaling/mean normalization):

- Compute mean: μj = 1/m Σ xj^(i)
- Replace each xj^(i) with xj^(i) - μj
- If different features on different scales (e.g., x1 = size of house, x2 = number of bedrooms), scale
  features to have comparable range of values
  + xj^(i) --> (xj^(i) - μj) / sj
    * sj can be the max() - min() value (range)
    * or sj is the standard deviation (commonly)

Principal Component Analysis (PCA) algorithm

- Reduce data from n-dimensions to k-dimensions
- Compute 'covariance matrix' Σ:

Σ-matrix = 1/m * Σ((x^(i)) * (x^(i)^T)) % call it Sigma

- Compute 'eigenvectors' of matrix Σ:

[U, S, V] = svd(Sigma); % Singular value decomposition

- This will return a n * n matrix
- We only need the U matrix (n * n)
- The columns will be the u vectors we want: u^(1), u^(2), ..., u^(n)
  + If we want it to k-dimensions, we take u^(1), u^(2), ..., u^(k) --> you take the first k columns/vectors

We want x C- R^n --> z C- R^k

- Ureduce (matrix) = n * k
- z = Ureduce^T * x % x is the n dimensional feature vector here, we can also do x^(i) to get z^(i)
  + Ureduce^T: k * n
  + x: n * 1
  + z: k * 1 --> z C- R^k

PCA algorithm Summary

- After mean normalization (ensure every feature has zero mean) and optionally feature scaling:

Sigma = 1/m Σ((x^(i)) * (x^(i)^T)) % Octave: Sigma = (1/m) * X' * X;
[U, S, V] = svd(Sigma); % n * n
Ureduce = U(:, 1:k)     % n * k --> grab all rows, up to k columns
z = Ureduce' * x        % k * n x n * 1

% Applying PCA

[Reconstruction from Compressed Representation]

z C- R --> x C- R^2

Xapprox = Ureduce * z % you get X approximately back (uncompressed)

[Choosing the Number of Principal Components]

Choosing k (number of principal components)
[1] - Average squared projection error: 1/m * Σ||x^(i) - xapprox^(i)||^2 % it tries to minimize the projection error between x and it's projection
[2] - Total variation in the data: 1/m * Σ||x^(i)||^2 % average length squared, on average, how far are my examples away from 0 or the origin

Typically, choose k to be the smallest value so that

[1] (1/m * Σ||x^(i) - xapprox^(i)||^2) / (1/m * Σ||x^(i)||^2) [2] <= 0.01 % 1%

'99% of variance is retained'

Another way to calculate this is by using the S matrix from [U, S, V] = svd(Sigma)

For given k:

     k       n
1 - (Σ Sii / Σ Sii) <= 0.01 % or drop the '1 -' & change to >= 0.99
    i=1     i=1

- Note that S is a diagonal matrix, only the diagonal is populated, rest is 0.
- Hence you only have, S11, S22, S33, etc. (Sii down to Snn).
- This is a quicker way to assess what value of k you need to retain 99% of the variance.

- Other common value is 0.05 (5%)

[Advice for Applying PC]

Bad use of PCA: to prevent overfitting

- Use z^(i) instead of x^(i) to reduce the number of featrues to k < n.
- Thus, fewer features, less likely to overfit (bad!).
- This might work OK, but is not a good way to address overfitting. Use regularization instead.

-- Week 9

== Anomaly Detection ==

-- Density Estimation --

[Problem Motivation]

Anomaly detection example

Aircraft engine features:
x1 = heat generated
x2 = vibration intensity
...

Dataset = {x^(1), x^(2), ..., x^(m)}

New engine: xtest

Is xtest anomalous?

Build a model: p(x), the probability of x

% ε is just some threshold
p(xtest) < ε % --> flag as anomaly --> the probability is very low to occur, hence anomaly
p(xtest) >= ε % --> looks OK

Fraud detection:
- x^(i) = features of user i's activities
% things like:
  % + x1 - how often does the user log in
  % + x2 - number of web pages visited
  % + x3 - number of transactions (e.g. number of posts)
  % + x4 - typing speed of user
- Model p(x) from data
- Identify unusal users by checking which have p(x) < ε % 'probability of x less than epsilon'
- Unusual != fraudulent, but can be

Manufacturing:
- Monitoring computers in a data center.
  + x^(i) = features of machine i
    * x1 = memory use
    * x2 = number of disk accesses/sec
    * x3 = CPU load
    * x4 = CPU load/network traffic
    * ...

[Gaussian Distribution]

Gaussian (Normal) Distribution

Say x C- R (x is a real number, 1-dimensional). If x is a distributed Gaussian with mean μ and variance σ^2.
- x ~ N(μ,σ^2) % how you write that x is normally N() distributed, with mean μ and variance σ^2
% ~ == 'distributed as'

p(x; μ,σ^2) = (1 / (2*pi^0.5) * σ) * exp(- (x - μ)^2 / 2σ^2) % exp() == e^-

σ = standard deviation
σ^2 = variance

Parameter Estimation

Dataset: {x^(1), x^(2), ..., x^(m)} where x^(i) C- R % x is a real number, 1-dim vector

x^(i) ~ N(μ,σ^2)

μ   = 1/m Σ x^(i)
σ^2 = 1/m Σ (x^(i) - μ)^2 % 1/(m-1) --> for samples, ML usually tends to use 1/m

[Algorithm]

Density estimation

Training set: {x^(1), ..., x^(m)}
Each example is x C- R^n

p(x) = p(x1;μ1,σ1^2) * p(x2;μ2,σ2^2) * p(x3;μ3,σ3^2) * ... * p(xn;μn,σn^2)

x1 ~ N(μ1,σ1^2)
x2 ~ N(μ2,σ2^2)
x3 ~ N(μ3,σ3^2)

Π = capital pi, represents the product of a set of values (as opposed to the sum: Σ)

 n
 Π (p(xj;(xj;μj,σj^2)))
j=1

 n
 Σ i = 1 + 2 + 3 + ... + n
 Π i = 1 * 2 * 2 * ... * n
i=1

Anomaly detection algorithm

1 - Choose features xi that you think might be indicative of anomalous examples.
2 - Fit paramters, μ1, ..., μn & σ1^2, ...., σn^2

μj   = 1/m Σ xj^(i) % j for every feature, up to μn
σj^2 = 1/m Σ (xj^(i) - μj)^2

Given a new example x, compute p(x):

p(x) = Π p(xj;μj,σj^2) = Π (1 / (2*pi^0.5) * σj) * exp(- (xj - μj)^2 / 2σj^2)

Anomaly if p(x) < ε

-- Building an Anomaly Detection System --

[Developing and Evaluating an Anomaly Detection System]

The Importance of real-number evaluation
- When developing a learning algorithm (choosing features, etc.), making
  decisions is much easier if we have a way of evaluating our learning algorithm.
- Assume we have some labeled data, of anomalous and non-anomalous examples
  (y = 0 if normal, y = 1 if anomalous).

Aircraft engines motivating example
- 10000 good (normal) engines
- 20 flawed engines (anomalous)

- Training Set: 6000 good engines (y = 0 as far as we know)
  + We use this set to get all μj,σj^2 for all features
  + We then have a model for p(x), using those values --> p(x) = p(x1;μ1,σ1^2) * p(x2;μ1,σ1^2) * .... * etc.
CV: 2000 good engines (y = 0), 10 anomalous (y = 1)
Test: 2000 good engines (y = 0), 10 anomalous (y = 1)

Algorithm evaluation

- Fit model p(x) on training set
- On a cross validation/test example x, predict:

y = { 1   if p(x) < ϵ  (anomaly)
    { 0   if p(x) >= ϵ (normal)

- Possible evaluation metrics:
  + TP, FP, FN, TN
  + Precision/Recall
    * F1-Score

Can also use CV set to choose parameter ϵ % for instance take the value of ϵ that maximizes the F1 score

% Classification accuracy is not a good measure because of skewed data (there are much more 0's than 1's)
% Hence, always predicting 0 will yield to high accuracy, since only 10 wrong out of 10010

Can also use CV set to choose parameter ϵ

- You evaluate the algorithm on the CV set, to select features, ϵ, etc.
- The final evaluation happens on the test set

[Anomaly Detection vs. Supervised Learning]

A - Anomaly Detection

- Very small number of positive examples (y = 1). 0-20 is common
- Large number of negative examples (y = 0).
- Many different 'types' of anomalies. hard for any algorithm to learn from positive examples
  what the anomalies look like; future anomalies may look nothing like any of the anomalous
  examples we've seen so far.

Applications:
- Fraud detection
- Manufacturing (e.g. aircraft engines)
- Monitoring machines in a data center

B - Supervised Learning

- Large number of positive and negative examples
- Enough positive examples for algorithm to get a sense of what positive examples are like, future
  positive examples likely to be similar to ones in training set.

Applications:
- Email spam classification
- Weather prediction (sunny/rainy/etc.)
- Cancer classification

[Choosing What Features to Use]

Non-Gaussian Features

- Sanity check: plot data in histogram to see if it is normally distributed.
  + hist()
- If your data is not normal, transform it to make it more Gaussian:
  1 - Take the log() of it: x1 --> log(x1)
  2 - x2: log(x2 + c)
    - x3: x3^c
      * Play around with c to make it as Gaussian as possible

Octave Demo:

size(x) % show num examples
hist(x,50) % give histogram with 50 bins, seems to have exponential distribution
hist(x.^.5,50) % root
hist(x.^.2,50)
hist(x.^.1,50)
hist(x.^.05,50)
xNew = x.^0.05 % this one looks more Gaussian
hist(log(x), 50) % also works pretty well
xNew = log(x)

Error analysis for anomaly detection

Want p(x) large for normal examples x.
     p(x) small for anomalous examples x.

Most common problem:
    p(x) is comparable (say, both large) for normal and anomalous examples.

- Take a close look at the example the algorithm got wrong, which hopefully inspires new features
  to take into consideration.

Monitoring computers in a data center

- Choose features that might take on unusally large or small values in the event of an anomaly.
  + x1 = memory use of computer
  + x2 = number of disk accesses/sec
  + x3 = CPU load
  + x4 = network traffic

x5 = CPU Load / Network Traffic

- Now x5 will take on a very large value if CPU Load is large but network traffic is not
  + You now have a feature that looks at the ratio of two other features

-- Multivariate Gaussian Distribution --

[Multivariate Gaussian Distribution]

Multivariate Gaussian (Normal) Distribution

x C- R^n. Do not model p(x1), p(x2), ..., etc. separately.
Model p(x) all in one go.
Parameters μ C- R^n, Σ C- R^n*n (covariance matrix)

p(x;μ,Σ) = 1 / ((2π)^n/2 * |Σ|^0.5) * exp(-1/2(x-μ)^T*Σ^-1(x-μ))

|Σ| = determinant of Σ % in Octave, calculate using det(Sigma)

μ = [0]
    [0]

Σ = [1   0.5] = x1 = [    std      correlation]
    [0.5 1  ] = x2 = [correlation      std    ]

- The more you increase the correlation, the more peaked it becomes, following the linear x=y line (x1=x2)
- The correlation can also be negative
  + Positive will be a hill from left below to upper right corner
  + Negative will be from left upper corner to right below corner, etc.
- Varying μ will move the peak/center of the distribution

Key advantage is it can model variables better when they are correlated.

[Anomaly Detection using the Multivariate Gaussian Distribution]

Parameter fitting:

Given training set {x^(1), x^(2, ..., x^m)}

1 - Fit model p(x) by setting:

μ = 1/m Σ x^(i)
Σ = 1/m Σ (x^(i) - μ) * (x^(i) - μ)^T

2 - Given a new example x, compute:

p(x) = 1 / ((2π)^n/2 * |Σ|^0.5) * exp(-1/2(x-μ)^T*Σ^-1(x-μ))

3 - Flag an anomaly if p(x < ϵ)

Relationship to Original Model

Original model:
p(x) = p(x1;μ1,σ1^2) * p(x2;μ2,σ2^2) * p(x3;μ3,σ3^2) * ... * p(xn;μn,σn^2)

Corresponds to multivariate Gaussian:

p(x;μ,Σ) = 1 / ((2π)^n/2 * |Σ|^0.5) * exp(-1/2(x-μ)^T*Σ^-1(x-μ)) % Σ^-1 == inverse of Σ

Where Σ = [σ1^2       0   ] --> all the correlations (values off the diagonal) are 0
          [    σ2^2       ]
          [        ...    ]
          [   0       σn^2]

- This means that like in the original model, the variations, etc. are axis-aligned.
- There are no models that are non-straight (following a correlation), such as bottom left corner to upper right corner.

Original Model vs. Multivariate Gaussian Model

A - Original Model

- Manually create features to capture anomalies where x1, x2 take unusual combinations of values (e.g. define x1/x2 ratio's, put in x3, etc.)
- Computationally cheaper (alternatively, scales better to large n)
- OK even if m (training set size) is small

B - Multivariate Gaussian

- Automatically captures correlations between features.
- Computationally more expensive
- Must have m > n, or else Σ is non-invertible % rule of thumb: m >= 10n
  + Another reason why Σ can be non-invertible is/are redundant features, e.g. x1 = x2 or x3 = x1 + x2

== Recommender Systems ==

-- Predicting Movie Ratings --

[Problem Formulation]

Example: Predicting Movie Ratings
- User rates movies using one to five (1-5) stars
  + We allow zero to five (0-5)

nu = number of users
nm = number of movies
r(i,j) = 1 if user j has rated movie i
y^(i,j) = rating given by user j to movie i % defined only if r(i,j) = 1

[Content Based Recommendations]

x0 = 1
x1 = correlation with being 'an action movie'
x2 = correlation with being 'a romance movie'

% We now have a feature vector:

x^(1) = [1  ] % intercept
        [0.9] % correlation with romance
        [0  ] % correlation with action

n = 2 (num features, without intercept)

For each user j, learn a parameter θ^(j) C- R^3.
Predict user j as rating movie i with (θ^(j))^T * x^(i) stars.

Problem formulation

r(i,j) = 1 if user j has rated movie i (0 otherwise)
y^(i,j) = rating by user j on movie i (if defined)

θ^(j) = parameter vector for user j
x^(i) = feature fector for movie i

For user j, movie i, predicted rating: (θ^(j))^T * x^(i) % simply inner product

m^(j) = number of movies rated by user j

To learn θ^(j):

                                                      n
min 1/2 * Σ (((θ^(j))^T * x^(i)) - y^(i,j))^2 + λ/2 * Σ (θk^(j))^2
θ^(j) i:r(i,j) = 1                                   k=1 % we do not regularize over the intercept (k=0); k = number of features
        % summing over all (i,j) values where r(i,j) = 1 (is present)

To learn θ^(1), θ^(2), ..., θ^(nu): % learn parameters for all users, so you just sum the above expression over all users!

                         nu                                                          nu  n
       min         1/2 * Σ        Σ        (((θ^(j))^T * x^(i)) - y^(i,j))^2 + λ/2 * Σ   Σ (θk^(j))^2
θ^(1), ... θ^(nu)       j=1  i:r(i,j) = 1                                           j=1 k=1

== J(θ^(1), ... θ^(nu))

- You get a separate parameter vector for each user, to make separate recommendations.
- You can use gradient descent to learn your parameter θ:

θk^(j) := θk^(j) - α * Σ ((θ^(j))^T * x^(i) - y^(i,j)) * xk^(i) % for k = 0
θk^(j) := θk^(j) - α * Σ ((θ^(j))^T * x^(i) - y^(i,j)) * xk^(i) + λθk^(j) % for k != 0

- k = number of features (e.g. k = 0 == intercept)
- To get the gradients here you use the chain rules!

-- Collaborative Filtering --

[Collaborative Filtering]

- Given x^(1), ..., x^(m) (and movie ratings), you can estimate θ^(1), ..., θ^(nu).
- Given θ^(1), ..., θ^(nu), you can estimate x^(1), ..., x^(m)

Procedure: Guess θ --> estimate x --> better estimate θ --> better x --> etc.

- The matrices are 'collaborating'.
- With parameters you learn features, and with features you learn parameters.

[Collaborative Filtering Algorithm]

- (i,j):r(i,j) = 1 == all user/movie pairs where r = 1
- Because we now learn our own features, we no longer need an intercept.

1. Initialize x^(1), ..., x^(m), θ^(1), ..., θ^(nu) to small random values.
2. Minimize J(x^(1), ..., x^(m), θ^(1), ..., θ^(nu)) using gradient descent
   (or an advanced optimization algorithm). E.g. for every j = 1, ..., nu
   & i = 1, ..., nm:

   % j:r(i,j) = 1
   xk^(i) := xk^(i) - α * (Σ (θ^(j)^T * x^(i) - y^(i,j) * θk^(j) + λxk^(i)))

   % i:r(i,j) = 1
   θk^(i) := θk^(i) - α * (Σ (θ^(j)^T * x^(i) - y^(i,j) * θk^(j) + λθk^(i)))

 3. For a user with parameters θ and a movie with (learned) features x, predict
    a star rating of θ^Tx (not yet rated movie)

-- Low Rank Matrix Factorization --

[Vectorization: Low Rank Matrix Factorization]

Finding related movies

- For each product i, we learn a feature vector x^(i) C- R^n.
  + x1 = romance
  + x2 = action
  + x3 = comedy
  + x4 = ...

How to find movie a related to movie b?
  + Small ||x^(a) - x^(b)|| --> movie a and b are similar!
  + Equivalent to: having a small distance between a and b.

5 most similar movies to movie i:
- Find the 5 movies b with the smallest ||x^(a) - x^(b)||.

[Implementation Detail: Mean Normalization]

After mean normalization you now predict:

(θ^(j))^T * x^(i) + μi

Now, if we don't have any information about a user (no ratings), we will
predict at the average, instead of predicting a score of 0.

-- Week 10

== Large Scale Machine Learning ==

-- Gradient Descent with Large Datasets --

[Learning With Large Datasets]

Machine learning and data
- Classify between confusable words.
  + E.g.: {to, two, too}, {then, than}
- For breakfast I ate ____ eggs.

'It is not who has the best algorithm that wins. It is who has the most data.'

[Stochastic Gradient Descent]

The other Gradient Descent algorithm can be defined as batch gradient descent,
since it sums over all training examples before taking a step. Stochastic Gradient
Descent however, updates the parameters after each individual training example, which
scales better with large datasets.

cost(θ,(x^(i), y^(i))) = 1/2(hθ(x^(i)-y^(i)))^2

Jtrain(θ) = 1/m Σ cost(θ,(x^(i), y^(i)))

1 - Randomly shuffle/re-order dataset
2 - Repeat {
  for i = 1, ..., m {
    θj := θj - α * (hθ(x^(i))-y^(i)) * xj^(i)
    % (for j = 0, ..., n) --> number of features
    % You calculate the error by taking the entire hθ(x^(i))-y^(i)
    % It shows θj * xj because you optimize every single parameter in your theta vector,
      % one by one in a loop, in one go in a vectorized implementation!
    % (hθ(x^(i))-y^(i)) * xj^(i) == partial derivative of cost(θ,(x^(i), y^(i))) w.r.t. θj
    }
}

- You make steps with every training example, instead of first summing over all training examples.
- With every training example you tweak the parameters.

[Mini-Batch Gradient Descent]

- Batch garidnet descent: use all m examples in each iteration
- Stochastic gradient descent: use 1 example in each iteration
- Mini-batch gradient descent: use b examples in each iteration
  + b = mini-batch size
    * Typical choice: b = 10, typical range b = 2-100

Example: b = 10
- (x^(i), y^(i), ... x^(i+9), y^(i+9))

                     i+9
θj := θj - α * 1/10 * Σ(hθ(x^(k))-y^(k)) * xj^(k)
% 10 = b             k=i

Say b = 10, m = 1000.

Repeat {
  for i = 1, 11, 21, 31, ..., 991 {
    θj := θj - α * 1/10 * Σ(hθ(x^(k))-y^(k)) * xj^(k)
    (for every j = 0, ..., n)
    }
}

Vectorization in mini-batch gradient descent allows for parallelization,
in contrast to stochastic gradient descent.

[Stochastic Gradient Descent Convergence]

- During learning, compute cost(θ,(x^(i), y^(i))) before updating θ using (x^(i), y^(i)).
- Every 1000 iterations (say), plot cost(θ,(x^(i), y^(i))) averaged over the last 1000 examples
  processed by the algorithm.

With Stochastic Gradient Descent your algorithm will oscillate around the global minimum at some point,
due to the learning rate α.
- Learning rate α is typically held constant.
- You can slowly decrease α over time if you want θ to converge.
  + E.g. α = constant1 / iterationNumber + constant2
    * As the algorithm runs, iterationNumber becomes larger and larger, hence α will become smaller and smaller

-- Advanced Topics

[Online Learning] % --> stochastic GD without saving data (training examples) + can adapt

Shipping service website where user comes, specifies origin and destination, you offer to
ship their package for some asking price and users sometimes choose to use your shipping service (y = 1),
sometimes not (y = 0).

Features x capture properties of user, of origin/destination and asking price. We want to learn
p(y = 1|x;θ) to optimize price (note: features include price).

Repeat forever {
  Get (x,y) corresponding to user.
  % x == origin/destination, price, etc.
  % y == used yes/no 1/0
  Update θ using (x,y):
  θj := θj - α * (hθ(x)-y) * xj % (j = 0, ..., n) --> this is a lot like stochastic gradient descent
}

- However, we do not store this training example, we only use it to optimize our parameters.
- This can adapt to changing user preferences
  + Say your pool of users changes, it will then slowly change according to their preferences.

Other online learning example:

Product search (learning to search):
- User searches for 'Android phone 1080p camera'
- Have 100 phones in store. Will return 10 results. % so every search will return 10 x,y pairs to train on again
  + x = features of phone
    * how many words in user query match name of phone
    * how many words in query match description, etc.
  + y = 1 if user clicks on link, y = 0 otherwise
    * Learn p(y = 1|x;θ) % learning predicted CTR

Other examples: Choosing special offers to show user; customized selection of news articles;
product recommendation; ...

[Map Reduce and Data Parallelism]

Map-reduce

Batch gradient descent:

Say m = 400 % in practice, think 400,000,000

                      400
θj := θj - α * 1/400 * Σ(hθ(x^(i))-y^(i)) * xj^(i) % This is centralised at one computer
                      i=1

Here we use a distributed system:

Machine 1: Use (x^(1),y^(1), ..., x^(100),y^(100))

           100
tempj^(1) = Σ (hθ(x^(i))-y^(i)) * xj^(i)
           i=1

Machine 2: Use (x^(101),y^(101), ..., x^(200),y^(200))

           200
tempj^(1) = Σ (hθ(x^(i))-y^(i)) * xj^(i)
          i=101

Machine 3: Use (x^(201),y^(201), ..., x^(300),y^(300))

etc.

Machine 4: Use (x^(301),y^(301), ..., x^(400),y^(400))

etc.

Combine:

θj := θj - α * 1/400 (tempj^(1) + tempj^(2) + tempj^(3) + tempj^(4)) % (j = 0, ..., n)

-- Week 11

== Photo OCR == % Optical Character Recognition

[Problem Description Pipeline]

1 - Text detection
2 - Character segmentation
3 - Character classification

Image --> Text Detection --> Character Segmentation --> Character Recognition

[Sliding Windows]

- Step-size / stride parameter
  + Usually a size of 1 px works best, however expensive (4 px can work too)
  + You slide by the x & y axis of the image
  + You also vary the slider size, but then resize it before feeding it to the classifer
    * To the format it recognizes
  + Expansion --> if a pixel is close to a 'white' pixel, make it white too (i.e. y = 1)
    * So you increase the area around the spots where you found text
  + Now you draw decision boundaries on all spots where the aspect ratio matches those of letters

Character Segmentation
- Classify images on being the midpoint of two characters
  + Now you can run this on everything the text detection stage brought up
  + Hence you now know where to 'cut' these areas into single characters

Character Recognition
- Now you feed all individual characters to another classifier to recognise characters.

[Getting Lots of Data and Artificial Data]

- Get a low bias (high variance) algorithm, train it on lots of data.

Artificial data synthesis for photo OCR

- Real data --> from real sources
- Synthetic data --> e.g. using letters from various fonts and putting them against various backgrounds

Synthesizing data by introducing distortions
- Get an imagine apply distortions, train on this to become more holistic
- Speech recognition:
  + Add audio on bad cellphone connection
  + Add audio on noisy background
    * Crowd
    * Machinery

- So the idea is you only collect one sample, but multiply it yourself by adding distortions.
- The distortions introduced should be representation of the type of noise/distortion in the test set.
  + Audio:
    * background noise, bad cellphone connection, etc.
- Usually does not help to add purely random/meaningless noise to your data.

Discussion on getting more data

1 - Make sure you have a low bias classifier before expending the effort (plot learning curves).
    + E.g. keep increasing the number of features/number of hidden units in neural network until
      you have a low bias classifier.
2 - 'How much work would it be to get 10x as much data as we curently have?'
    + Artificial data synthesis
    + Collect/label it yourself % this can be surprisingly little work
    + 'Crowd source' (e.g. Amazon Mechical Turk)
      * Hire people on the web to label large datasets
        - Labeler reliability can be a problem

[Ceiling Analysis: What Part of the Pipeline to Work on Next]

Image --> Text Detection --> Character Segmentation --> Character Recognition

- Example: Feed 100 percent accurate data to:
  + Overall Accuracy: 73%
  + Text Detection --> Measure overall accuracy: 89%
  + Text Detection & Character Segmentation --> Measure overall accuracy: 90%
  + Text Detection, Character Segmentation & Character Recognition: 100%
    * Now measure the incremental steps between them to assess what to work on next (biggest jump):
      - 89 - 73 = 16%
      - 90 - 89 = 1%
      - 100 - 90 = 10%
        + So, first work on Text Detection (+16) then Character Recognition (+10)
          * Character Segmentation (+1) may not be worth it

[Conclusion]

Supervised Learning
- Linear regression, logistic regression, neural networks, SVMs

Unsupervised Learning
- K-means, PCA, Anomaly detection

Special applications/special topics
- Recommender systems, large scale machine learning.

Advice on building a machine learning system
- Bias/variance, regularization; deciding what to work on next: evaluation of learning
  algorithms, learning curves, error analysis, ceiling analysis.
