import numpy as np
import random

function_evaluations = 0

# Problem 1 objective function
def main_obj_1(x):
    global function_evaluations
    function_evaluations += 1
    x1, x2 = x
    return (x1 - 10)**3 + (x2 - 20)**3

# Problem 2 objective function (converted to minimization)
def main_obj_2(x):
    global function_evaluations
    function_evaluations += 1
    x1, x2 = x
    return -(np.sin(2*np.pi*x1)**3 * np.sin(2*np.pi*x2)) / (x1**3 * (x1 + x2))

# Problem 3 objective function
def main_obj_3(x):
    global function_evaluations
    function_evaluations += 1
    return x[0] + x[1] + x[2]  # x1 + x2 + x3

# Problem 1 constraints
def g1_prob1(x):
    global function_evaluations
    function_evaluations += 1
    x1, x2 = x
    return (x1 - 5)**2 + (x2 - 5)**2 - 100

def g2_prob1(x):
    global function_evaluations
    function_evaluations += 1
    x1, x2 = x
    return 82.81 - (x1 - 6)**2 - (x2 - 5)**2  #(converted to >= form by multiplying by -1)

# Problem 2 constraints (converted to >= form by multiplying by -1)
def g1_prob2(x):
    global function_evaluations
    function_evaluations += 1
    x1, x2 = x
    return -(x1**2 - x2 + 1)

def g2_prob2(x):
    global function_evaluations
    function_evaluations += 1
    x1, x2 = x
    return -(1 - x1 + (x2 - 4)**2)

# Problem 3 constraints (converted to >= form by multiplying by -1)
def g1_prob3(x):
    global function_evaluations
    function_evaluations += 1
    return -(-1 + 0.0025*(x[3] + x[5]))

def g2_prob3(x):
    global function_evaluations
    function_evaluations += 1
    return -(-1 + 0.0025*(-x[3] + x[4] + x[6]))

def g3_prob3(x):
    global function_evaluations
    function_evaluations += 1
    return -(-1 + 0.01*(-x[5] + x[7]))

def g4_prob3(x):
    global function_evaluations
    function_evaluations += 1
    return -(100*x[0] - x[0]*x[5] + 833.33252*x[3] - 83333.333)

def g5_prob3(x):
    global function_evaluations
    function_evaluations += 1
    return -(x[1]*x[3] - x[1]*x[6] - 1250*x[3] + 1250*x[4])

def g6_prob3(x):
    global function_evaluations
    function_evaluations += 1
    return -(x[2]*x[4] - x[2]*x[7] - 2500*x[4] + 1250000)

def get_constraints(problem_num):
    if problem_num == 1:
        return [g1_prob1, g2_prob1]
    elif problem_num == 2:
        return [g1_prob2, g2_prob2]
    else:
        return [g1_prob3, g2_prob3, g3_prob3, g4_prob3, g5_prob3, g6_prob3]

def get_objective(problem_num):
    if problem_num == 1:
        return main_obj_1
    elif problem_num == 2:
        return main_obj_2
    else:
        return main_obj_3

def penalty_function(x, R, problem_num):
    global function_evaluations
    constraints = get_constraints(problem_num)
    penalty = 0
    for g in constraints:
        penalty += R * max(0, -g(x))**2
    return penalty

def P(x, R, problem_num):
    global function_evaluations
    obj = get_objective(problem_num)
    return obj(x) + penalty_function(x, R, problem_num)

def numerical_gradient(f, x, R, h=1e-6):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_step_forward = np.array(x, dtype=float)
        x_step_backward = np.array(x, dtype=float)
        x_step_forward[i] += h
        x_step_backward[i] -= h
        grad[i] = (f(x_step_forward, R) - f(x_step_backward, R)) / (2 * h)
    return grad

def numerical_hessian(f, x, R, h=1e-6):
    hessian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            x_ij = np.array(x, dtype=float)
            x_ij[i] += h
            x_ij[j] += h
            f_pp = f(x_ij, R)
            
            x_ij[j] -= 2*h
            f_pm = f(x_ij, R)
            
            x_ij[i] -= 2*h
            x_ij[j] += 2*h
            f_mp = f(x_ij, R)
            
            x_ij[j] -= 2*h
            f_mm = f(x_ij, R)
            
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)
    return hessian

def bounding_phase_method(f, a, b, delta, max_iter=100, max_retries=10):
    retry = 0
    for retry in range(max_retries):
        x = np.random.uniform(a, b)  # Random initial guess in the interval
        f_x_minus = f(x - abs(delta))
        f_x = f(x)
        f_x_plus = f(x + abs(delta))

        if f_x_minus >= f_x >= f_x_plus:
            delta = abs(delta)
        elif f_x_minus <= f_x <= f_x_plus:
            delta = -abs(delta)
        else:
            retry+=1
            continue  # Retry with a new random guess

        k = 0
        while k < max_iter:
            x_next = x + 2**k * delta
            if f(x_next) < f_x:
                x = x_next
                f_x = f(x)
                k += 1
            else:
                x_high = x_next
                x_low = x - 2**(k-1) * delta
                return x, x_low, x_high
        retry+=1
    raise ValueError("Bounding Phase Method failed to find a suitable interval within max retries.")

def derivative(fd, x, h=1e-6):
    return (fd(x + h) - fd(x - h)) / (2*h)

def second_derivative(fd, x, h=1e-6):
    return (fd(x + h) - 2*fd(x) + fd(x - h)) / (h**2)

def newton_raphson_method(f, x_low, x_high, epsilon, max_iter=100):
    x = np.random.uniform(x_low, x_high) # Start with a random point in the interval
    k = 0

    while k < max_iter:
        f_prime = derivative(f, x)
        f_double_prime = second_derivative(f, x)

        if abs(f_double_prime) < 1e-14: 
            x = random.uniform(x_low, x_high)  # Reinitialize x randomly
        else:
            x_next = x - (f_prime / f_double_prime)
            f_prime_next = derivative(f, x_next)

            if abs(f_prime_next) < epsilon:
                return x_next
            x = x_next
        k += 1
    return x

def unidirectional_search(f, a, b, delta, epsilon):
    alpha, alpha_low, alpha_high = bounding_phase_method(f, a, b, delta)
    return newton_raphson_method(f, alpha_low, alpha_high, epsilon)

def calculate_alpha_bounds(x_curr, s_k, a, b):
    alpha_min = float('-inf')
    alpha_max = float('inf')
    
    for i in range(len(x_curr)):
        if abs(s_k[i]) < 1e-3:  # Skip if direction component is essentially zero
            continue
            
        if s_k[i] > 0:
            alpha_max = min(alpha_max, (b[i] - x_curr[i]) / s_k[i])
            alpha_min = max(alpha_min, (a[i] - x_curr[i]) / s_k[i])
        else:
            alpha_max = min(alpha_max, (a[i] - x_curr[i]) / s_k[i])
            alpha_min = max(alpha_min, (b[i] - x_curr[i]) / s_k[i])
    
    # Ensure we have a valid range (if constraints allow movement)
    if alpha_min > alpha_max:
        alpha_min = 0
        alpha_max = 0
    
    # If bounds are infinite, set reasonable defaults
    if alpha_min == float('-inf'):
        alpha_min = 0
    if alpha_max == float('inf'):
        alpha_max = 1.0
        
    return alpha_min, alpha_max

def marquardt_method(f, a, b, x, delta, epsilon, epsilon_1, lambda_0, R, M=100):
    k = 0
    lambda_k = lambda_0
    x_best = x.copy()
    f_best = f(x, R)
    
    while k < M:
        grad = numerical_gradient(f, x, R)
        
        if np.linalg.norm(grad) <= epsilon_1:
            return x_best
            
        H = numerical_hessian(f, x, R)
        H_lambda = H + lambda_k * np.eye(len(x))
        
        try:
            s_k = -np.linalg.solve(H_lambda, grad)
        except np.linalg.LinAlgError:
            lambda_k *= 10
            continue
            
        norm_s_k = np.linalg.norm(s_k)
        if norm_s_k > epsilon:
            s_k = s_k / norm_s_k
            
        def f_alpha(alpha):
            x_new = x + alpha * s_k
            #x_new = np.clip(x_new, a, b)
            return f(x_new, R)
        
        # Calculate appropriate alpha bounds based on variable bounds
        alpha_min, alpha_max = calculate_alpha_bounds(x, s_k, a, b)
            
        try:
            alpha_k = unidirectional_search(f_alpha, alpha_min, alpha_max, delta, epsilon)
        except ValueError:
            alpha_k = min(0.1, alpha_max)  # Use small step if line search fails
            
        #x_new = np.clip(x + alpha_k * s_k, a, b)
        x_new = x + alpha_k * s_k
        f_new = f(x_new, R)
        
        if f_new < f_best:
            x_best = x_new.copy()
            f_best = f_new
            lambda_k /= 2
        else:
            lambda_k *= 2
            
        x = x_new
        k += 1
        
    return x_best

def bracket_operator_penalty(f, a, b, x, delta, epsilon, epsilon_1, lambda_0, R0, epsilon_2, c, problem_num, maxiter=100):
    R = R0
    x_current = x.copy()
    k = 0
    
    while k < maxiter:
        x_new = marquardt_method(lambda x, R: f(x, R, problem_num), a, b, x_current, delta, epsilon, epsilon_1, lambda_0, R)
        
        if abs(P(x_new, R, problem_num) - P(x_current, R, problem_num)) <= epsilon_2:
            print(f"Converged in {k} iterations.")
            return x_new
            
        R *= c
        x_current = x_new
        k += 1
        
    print(f"Warning: Maximum iterations ({maxiter}) reached.")
    return x_current

def main():
    global function_evaluations
    
    print("\nAvailable test problems:")
    print("1. Original problem (2 variables)")
    print("2. sin³(2πx₁)sin(2πx₂)/(x₁³(x₁+x₂)) problem (2 variables)")
    print("3. Linear objective with nonlinear constraints (8 variables)")
    
    problem_num = int(input("\nSelect problem number (1-3): "))
    
    if problem_num == 1:
        n_vars = 2
        default_bounds = [(13, 20), (0, 4)]
    elif problem_num == 2:
        n_vars = 2
        default_bounds = [(0, 10), (0, 10)]  # Avoiding x₁=0 due to division
    else:
        n_vars = 8
        default_bounds = [(100, 10000), (1000, 10000), (1000, 10000),
                         (10, 1000), (10, 1000), (10, 1000),
                         (10, 1000), (10, 1000)]
    
    delta = 0.01
    epsilon = 1e-3
    epsilon_1 = 1e-3
    epsilon_2 = 1e-3
    lambda_0 = 100
    R0 = 0.1
    c = 10
    
    use_default = input(f"\nUse default bounds for problem {problem_num}? (y/n): ").lower() == 'y'
    
    if use_default:
        a = np.array([bound[0] for bound in default_bounds])
        b = np.array([bound[1] for bound in default_bounds])
    else:
        a = []
        b = []
        for i in range(n_vars):
            lower_bound = float(input(f"Enter the lower bound for variable {i+1}: "))
            upper_bound = float(input(f"Enter the upper bound for variable {i+1}: "))
            a.append(lower_bound)
            b.append(upper_bound)
        a = np.array(a)
        b = np.array(b)
    
    for i in range(10):
        print(f"\n--- Run {i+1} ---")
        function_evaluations = 0
        
        x_0 = np.array([np.random.uniform(a[j], b[j]) for j in range(n_vars)])
        print(f"Initial value of x0: {x_0}")
        
        try:
            result = bracket_operator_penalty(P, a, b, x_0, delta, epsilon, epsilon_1, 
                                           lambda_0, R0, epsilon_2, c, problem_num)
            print(f"Optimized parameters: {result}")
            obj = get_objective(problem_num)
            print(f"Function value at optimized parameters: {obj(result)}")
            print(f"No. of Function Evaluations: {function_evaluations}")
            
            # Print constraint values
            constraints = get_constraints(problem_num)
            for j, g in enumerate(constraints, 1):
                print(f"Constraint g{j}: {g(result)}")
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            continue

if __name__ == "__main__":
    main()