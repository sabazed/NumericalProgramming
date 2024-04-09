import matplotlib.pyplot as plt
import ExtractUtils
import Lagrange
import CubicSpline
import LeastSquares

### CP1
### Main problem was to implement different approximation methods and test it against closed curves
### I have implemented them within their classes, as you can create their objects and just like a function,
### execute .apply() method for each of them, while providing x and y points to each approximation
### during construction (and additional parameters if needed). The objects themselves will generate all
### necessary attributes and methods to let the user call apply() only and return a tuple of coordinates

### Lagrange interpolation, least squares method, and cubic splines are all numerical techniques for approximating 
### closed curves. Lagrange interpolation constructs a polynomial that passes through given data points, potentially 
### leading to oscillations. The least squares method minimizes the sum of squared errors, providing a flexible approach 
### but sensitive to outliers. Cubic splines ensure smoothness by connecting data points with cubic polynomials, maintaining 
### continuity and differentiability. Each method has its strengths and limitations, with the choice depending on the 
### specific characteristics of the data and the desired curve approximation.

### # Accuracy and Smoothness:
### Lagrange Interpolation: High accuracy but prone to oscillations.
### Least Squares Method: Flexible but may lack smoothness, especially with outliers.
### Cubic Splines: Balances accuracy and inherent smoothness.
### # Flexibility and Computational Complexity:
### Lagrange Interpolation: Limited flexibility; computation can be intensive for high degrees.
### Least Squares Method: Highly flexible, computational cost depends on basis functions.
### Cubic Splines: Moderately flexible, generally computationally efficient.
### # Interpolation Range and Implementation:
### Lagrange Interpolation: Confined to data points; straightforward implementation.
### Least Squares Method: Can extrapolate but sensitive to outliers; implementation complexity varies.
### Cubic Splines: Constrained to data hull; relatively simple implementation.

if __name__ == "__main__":

    # Utility to extract curve info
    curveUtils = ExtractUtils(10, False) # Set parameter to true when using path from local storage
    path = "https://i.pinimg.com/1200x/09/6b/9f/096b9f21d164aa34a980c85b8a5994b4.jpg"
    # Get x and y points
    x, y = curveUtils.get_curve_points(path)

    # Get Lagrange interpolated points
    lagrange = Lagrange(x, y)
    lagrange_x, lagrange_y = lagrange.apply()

    # Get Cubic Spline points
    cubicSpline = CubicSpline(x, y)
    spline_x, spline_y = cubicSpline.apply()

    # Get Least Square points
    leastSquares = LeastSquares(x, y, 20) # Adjust degree for better results
    lst_sqr_x, lst_sqr_y = leastSquares.apply()

    # Plot the curve points and all approximation results
    plt.figure()
    plt.axis('equal')
    plt.title('Approximation Comparison')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(x, y, label='Points', color='orange', marker='o', linewidth=2) 
    plt.plot(lagrange_x, lagrange_y, label='Lagrange Interpolation', color='red', linewidth=4)
    plt.plot(spline_x, spline_y, label='Cubic Spline Interpolation', color='green', linewidth=2)
    plt.plot(lst_sqr_x, lst_sqr_y, label='Least Square', color='blue', linewidth=1.5)
    plt.legend()
    plt.show()