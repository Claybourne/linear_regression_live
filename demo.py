from numpy import *
# y = mx + b
# m is slope, b is y-intercept
# for every Y data point, see how far on Y axis it is from line....add them all up, then square to make positive, then get average
# Error (m,b) = 1/N SIGMA_SUM (y-(mx +b))^2
# we want to minimise this value each time, to improve our guess....to make our gradient match the data
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    # starting point for our gradient
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # for every point we have we will calculate the partial derivative to give us a driection to go for b and m
        # so we can go down the gradient to find the minimum error
        # http://mathinsight.org/image/partial_derivative_as_slope 
        # df/dx (a,b) to give us the slope of the line
        
        # partial derivative for b
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        # partial derivative for b
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    # update our b an m values with our partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    #perform gradient descent
    for i in range(num_iterations):
        #update b and m witht he new more accurate b and m by performing this gradient step, then return the optimal b and m
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    #Step 1 - collect our data (use the numpy method)
    points = genfromtxt("data.csv", delimiter=",")
    
    #Step 2 - define the hyper parameters, how fast shoulf the model converge
    
    # Start the Y intercept b at 0 and the slope  m at 0 then move this straight line slope to match our required linear slope, witht he smallest average error (distance of points from the line).
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print "Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points))
    print "Running..."
    #now do the gradient descent
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print "After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points))

    # our main function
if __name__ == '__main__':
    run()
