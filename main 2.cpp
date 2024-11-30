#include <bits/stdc++.h>
#include <numeric> // For std::inner_product
using namespace std;

int func_eval = 0;


// Function pointer to hold the address of the objective functions
double (*fun_ptr)(const vector<double>&,double&);  // using fun_ptr for calling the penalty function

//functions
double c1(const vector<double> &x,double& R)
{
    double c = 0;
    double g1 = pow((x[0]-5),2) + pow((x[1]-5),2) - 100;
    double g2 = 82.81 - pow((x[0]-6),2) - pow((x[1]-5),2);
    if(g1 < 0)
    {
        c += pow(g1,2);
    }
    if(g2 < 0)
    {
        c += pow(g2,2);
    }
    c = c*R;
    return c;
}
double f1(const vector<double> &x,double& R){
    func_eval++;
    double fun = 0;
    fun = pow((x[0] - 10), 3) + pow((x[1] - 20), 3);
    double p = c1(x,R);
    fun = fun + p;
    return fun;
}

double c2(const vector<double> &x,double R)
{
    double c =0;
    double g1 = pow(x[0],2) - x[1] + 1;
    double g2 = 1 - x[0] + pow((x[1] - 4),2);
    if(g1 > 0)
    {
        c += pow(g1,2);
    }
    if(g2 > 0)
    {
        c += pow(g2,2);
    }
    c = c*R;
    return c;
}
double f2(const vector<double>& x,double& R){
    func_eval++;
    double fun = 0;
    double pi = M_PI;
    fun = pow(sin(2*pi*x[0]),3);
    fun = fun * sin(2*pi*x[1]);
    fun /= pow(x[0],3);
    fun /= (x[0] + x[1]);
    double p = c2(x,R);
    fun = -(fun) + p;
    double f = fun;
    return f;
}

double  c3(const vector<double>& x,double& R)
{ 
    double c =0;
    double g1 = -1 + 0.0025*(x[3] + x[5]);
    if(g1 > 0)
    {
        c += pow(g1,2);
    }
    double g2 = -1 + 0.0025*(-x[3] + x[4] + x[6]);
    if(g2 > 0)
    {
        c += pow(g2,2);
    }
    double g3 = -1 + 0.01*(-x[5] + x[7]);
    if(g3 > 0)
    {
        c += pow(g3,2);
    }
    double g4 = 100*x[0] - (x[0]*x[5]) + 833.33252*x[3] - 83333.333;
    if(g4 > 0)
    {
        c += pow(g4,2);
    }
    double g5 = (x[1] * x[3]) - (x[1] * x[6]) - 1250 * x[3] +1250*x[4];
    if(g5 > 0)
    {
        c += pow(g5,2);
    }
    double g6 = (x[2] * x[4]) - (x[2] * x[7]) - 2500*x[4] + 1250000;
    if(g6 > 0)
    {
        c += pow(g6,2);
    }
    c = c* R;
    return c;
}

double f3(const vector<double>&x,double &R)
{
    func_eval++;
    double fun = 0;
    fun = x[0] + x[1] + x[2];
    double p = c3(x,R);
    fun += p;
    return fun;
}


// function without penalised parameters.
double functionvalue(vector<double>&x,int choice)
{
    if(choice == 1)
    {
        return pow((x[0] - 10), 3) + pow((x[1] - 20), 3);
    }
    else if(choice == 2)
    {
    double fun = 0;
    double pi = 3.14159265358979323846;
    fun = pow((sin(2*pi*x[0])),3) * sin(2*pi*x[1]);
    fun /= pow(x[0],3);
    fun /= (x[0] + x[1]);
        return fun;
    }
    else if(choice == 3)
    {
        return x[0] + x[1] + x[2];
    }
    return 0.0;
}

// Function to choose which function to use
void select_function(int choice) {
    switch (choice) {
        case 1:
            fun_ptr=f1;
            break;
        case 2:
            fun_ptr = f2;
            break;
        case 3:
            fun_ptr = f3;
            break;
        default:
            cerr << "Invalid choice." << endl;
            fun_ptr = f1;
            break;

    }
}


// Finding gradient of a function
vector<double> gradient(const vector<double> &x,double &R)
{
    int n = x.size();
    vector<double> grad(n, 1.0);
    
      
    double h = 1e-6;
    for (int i = 0; i < n; ++i)
    {
      vector<double>  fplus = x;
        fplus[i] = fplus[i]+ h;
    vector<double> fminus = x;
        fminus[i] =fminus[i]- h;
        // gradient using central difference method
        grad[i] = (fun_ptr(fplus,R) - fun_ptr(fminus,R)) / (2 * h);
    }
    return grad;
}

// function to compute Hessian matric  numerically (central difference method)
vector<vector<double>> hessian(const vector<double> &x,double &R)
{
    int d = x.size();
    double h = 1e-6;
    vector<vector<double>> H(d, vector<double>(d, 0.0)); // Hessian matrix
    vector<double> x_plus_h = x;
    // Compute second-order partial derivatives
    for (int i = 0; i < d; i++)
    {
        for (int j = 0; j < d; j++)
        {
           
                // Off-diagonal elements: second derivative with respect to different variables
                x_plus_h[i] += h;
                x_plus_h[j] += h;
                double f_plusplus = fun_ptr(x_plus_h,R);
                 
                x_plus_h[j] -= 2 * h;
                double f_plusminus = fun_ptr(x_plus_h,R);  
        
                 
                x_plus_h[i] -= 2 * h;
                x_plus_h[j] += 2 * h;
                double f_minusplus = fun_ptr(x_plus_h,R);
               
                x_plus_h[j] -= 2 * h;
                double f_minusminus = fun_ptr(x_plus_h,R);
                
                H[i][j] = (f_plusplus - f_plusminus - f_minusplus + f_minusminus) / (4 * h * h);

            }
        }
    return H;
}

//Helper function to add lambda * Identity to Hessian
vector<vector<double>> add_lambda_identity(const vector<vector<double>> &H, double lambda)
{
    int n = H.size();
    vector<vector<double>> H_lambda = H;

    for (int i = 0; i < n; ++i)
    {
        H_lambda[i][i] += lambda;
    }

    return H_lambda;
}

// Helper function to compute inverse of a matrix (using Gaussian elimination here for simplicity)
void getCofactor(const vector<vector<double>> &A, vector<vector<double>> &temp, int p, int q, int n)
{
    int i = 0, j = 0;

    // Looping for each element of the matrix
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            // Copying into the temporary matrix only those elements which are not in the given row and column
            if (row != p && col != q)
            {
                temp[i][j++] = A[row][col];

                // Row is filled, move to next row
                if (j == n - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
}

// Function to calculate determinant of matrix A of size n
double determinant(const vector<vector<double>> &A, int n)
{
    double D = 0; // Initialize result

    // Base case : if matrix contains single element
    if (n == 1)
        return A[0][0];

    vector<vector<double>> temp(n, vector<double>(n)); // To store cofactors

    int sign = 1; // To store sign multiplier

    // Iterate for each element of first row
    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        getCofactor(A, temp, 0, f, n);
        D += sign * A[0][f] * determinant(temp, n - 1);

        // terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
}

// Function to get adjoint of matrix A of size n
void adjoint(const vector<vector<double>> &A, vector<vector<double>> &adj)
{
    int n = A.size();
    if (n == 1)
    {
        adj[0][0] = 1;
        return;
    }
    int sign = 1;
    vector<vector<double>> temp(n, vector<double>(n));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // Get cofactor of A[i][j]
            getCofactor(A, temp, i, j, n);

            // Sign of adj[j][i] positive if sum of row and column indexes is even.
            sign = ((i + j) % 2 == 0) ? 1 : -1;

            // Interchange rows and columns to get the transpose of the cofactor matrix
            adj[j][i] = (sign) * (determinant(temp, n - 1));
        }
    }
}

// Function to calculate inverse of matrix A
vector<vector<double>> inverse(const vector<vector<double>> &A)
{
    int n = A.size();
    double det = determinant(A, n);
    if (det == 0)
    {
        cout << "Singular matrix, can't find its inverse" << endl;
        return A;
    }
    vector<vector<double>> adj(n, vector<double>(n));
    adjoint(A, adj);
    vector<vector<double>> inv(n, vector<double>(n));
    // Inverse of matrix is adjoint divided by determinant
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv[i][j] = adj[i][j] / det;

    return inv;
}
// To multiply two vectors
vector<double> matrix_vector_multiply(const vector<vector<double>> &matrix, const vector<double> &vec)
{
    int n = vec.size();
    vector<double> result(n, 0.0); // Initialize the result as a nx1 vector with zero values

    for (int i = 0; i < n; ++i)
    {
        // Multiply each row of the matrix by the vector and sum up the products
        for (int j = 0; j < n; ++j)
        {
            result[i] += matrix[i][j] * (-1 * vec[j]); 
        }
    }

    return result;
}


pair<double, double> bounding_phase( function<double(double)> fun,double a, double b, double epsilon)
{
    int m=100;//max_iterations
    double delta = 0.5;
    double x,x_new,prev;
    double f_minus_delta, f_plus_delta, f_x;
    int k = 0;
   
    //cout<<"a1:"<<a<<" "<<b<<endl;
     
    while (1)
    {
        //cout<<"b"<<endl;
         srand(static_cast<unsigned int>(time(NULL)));
        x = a + (b - a) * ((double)rand() / RAND_MAX);
        //cout<<"x "<<x<<endl;
        f_minus_delta = fun(x - abs(delta));
        f_x = fun(x);
        f_plus_delta = fun(x + delta);
        if (f_minus_delta >= f_x && f_x >= f_plus_delta)
        {
            delta = abs(delta);
            break;
        }
        else if (f_minus_delta <= f_x && f_x <= f_plus_delta)
        {
            delta = -abs(delta);
            break;
        }
        
          this_thread::sleep_for(chrono::seconds(1));
    }
    // if(cnt>=100)
    // {
        
    // }
    //prev = x;
    x_new = x + delta;
    double f_new=fun(x_new);
    while (f_new <= f_x && x_new >= a && x_new <= b) {

        if(x_new < a || x_new >b)
        {
            break;
        }
		k++;
		prev = x;
        x = x_new;
		x_new = prev + (pow(2,k)) * delta;
		f_new=fun(x_new);
		f_x=fun(x);

	}
    if(x_new < a)
    {
        x_new =a;
    }
    if(x_new > b)
    {
        x_new = b;
    }
    return  {min(prev,x_new),max(prev,x_new)};
}


  
    


//Golden Section Method
double Golden_section(function<double(double)> fun,double a,double b,double e) {
	double aw,bw,lw,w1,w2,x,f1,f2;
	aw=0;
	bw=1;
	lw=bw-aw;
	int k=1;
	w1= aw + 0.618*lw;
	w2= bw - 0.618*lw;
	f1=fun((w1*(b-a))+a);
	f2=fun((w2*(b-a))+a);
  
	while(lw>e) {
		k+=1;
		if(f2>f1) {
			aw=w2;
			w2=w1;
			lw=bw-aw;
			w1= aw + 0.618*lw;
			f2=f1;
			f1=fun(w1*(b-a)+a);
		
				}
		else if(f1>f2) {
			bw=w1;
			w1=w2;
			lw=bw-aw;
			w2= bw - 0.618*lw;
			f1=f2;
			f2=fun(w2*(b-a)+a);
		
			
		}
		else  {
			aw=w2;
			bw=w1;
			lw=bw-aw;
			w1= aw + 0.618*lw;
			w2= bw - 0.618*lw;
			f1=fun(w1*(b-a)+a);
			f2=fun(w2*(b-a)+a);
		
		}
	}
	double midpoint = (aw+bw)/2;
	double ans=midpoint*(b-a)+a;
   return ans;

}

// normalising the vector to get unit vector
vector<double> normalise(vector<double>&A)
{
    
    
  for(int i=0;i<A.size();i++)
        {
            double m1 = sqrt(inner_product(A.begin(), A.end(), A.begin(), 0.0));
            if(m1!=0){
            A[i] = A[i] / m1;
        }}
            
        
        return A;
}
// UNI SEARCH METHOD
double uni_search_alpha(function<double(double)> fun, double delta, double epsilon, double a, double b)
{
   // cout<<"1-"<<endl;
    pair<double, double> interval = bounding_phase(fun, a, b, delta);
    //cout<<"4-"<<endl;
    return Golden_section(fun, interval.first, interval.second, epsilon);
}

bool check_linear_dependence(const vector<double> &s_k, const vector<double> &s_k_1)
{
    double y = inner_product(s_k.begin(), s_k.end(), s_k_1.begin(), 0.0);
    return abs(abs(y) - 1) < 1e-14;

}

//Marquardt function 
vector<double> Marquardt(vector<double> &x,vector<pair<double,double>>bound ,double& R,double& e1)
{
    int n = x.size();
    int M = 100;// no of max interations
    int k = 0; //no of iterations
    double epsilon = 1e-3, delta = 0.001;
    double lambda = pow(10, 3);
    vector<double> x_new(n, 0.0);//new x;
     vector<double> Sk_prev;
     
     vector<double> x_best=x;
  while (k<M)
  {
      //cout<<"1"<<endl;
        
        vector<double> grad = gradient(x,R); // calculated gradient
        double norm = sqrt(inner_product(grad.begin(), grad.end(), grad.begin(), 0.0));// calculating norm

        if (norm <= epsilon)
        {
           return x;
        }
        vector<vector<double>> H = hessian(x,R);
        vector<vector<double>> H_lambda_I = add_lambda_identity(H, lambda); // hessian +lambda*identity
        vector<vector<double>> Final_inverse = inverse(H_lambda_I);
         if (Final_inverse.empty()) {
            throw runtime_error("Hessian matrix is singular and cannot be inverted.");
        }
        vector<double> Sk = matrix_vector_multiply(Final_inverse, grad);
         
       Sk = normalise(Sk);
         if( !Sk_prev.empty() && check_linear_dependence(Sk,Sk_prev)){
         
            x=x_new;
            continue;
             
         }
         
      Sk_prev=Sk;
     
      //Finding the alpha
        double alpha_min = -DBL_MAX;
         double alpha_max = DBL_MAX;
       
        for (int i = 0; i < x.size(); i++)
        {
            if(abs(Sk[i])< 1e-3){
                continue;
            }
            
             if (Sk[i] > 0) {
            // Calculate maximum alpha such that x + alpha * d <= x_max
            alpha_max = min(alpha_max, (bound[i].second - x[i]) / Sk[i]);
            alpha_min = max(alpha_min, (bound[i].first - x[i]) / Sk[i]);
        } else if (Sk[i] < 0) {
            // Calculate maximum alpha such that x + alpha * d >= x_min
            alpha_max = min(alpha_max, (bound[i].first - x[i]) / Sk[i]);
            alpha_min = max(alpha_min, (bound[i].second- x[i]) / Sk[i]);
        }
        }
        
        
        
        
        double low_alpha=alpha_min;
        double high_alpha=alpha_max;
        
        auto obj_fun_val = [&x,&R, &Sk](double alpha)-> double
        {
            // cout<<"obj"<<endl;
            int n = x.size();
            vector<double> temp_x_new(n, 0.0);
            vector<double>s = Sk;
            for (int i = 0; i < n; i++)
            {
                temp_x_new[i] = x[i] + alpha * s[i];
            }
            // func_eval++;
            return fun_ptr(temp_x_new,R);
        };
        
        double alpha_k = uni_search_alpha(obj_fun_val, delta, epsilon, low_alpha,high_alpha);
        
        for (double i = 0; i < n; i++)
        {
            x_new[i] = x[i] + alpha_k * Sk[i];
        }
        double f_new=fun_ptr(x_new,R);
        double f_x=fun_ptr(x,R);
        if (f_new < f_x)
        {
          x_best = x_new;
            //x = x_new;
             f_x=f_new;
              lambda = lambda / 2;
          
        }
        else 
        {
            lambda = 2 * lambda;
        }
       x = x_new;
         k += 1;
        
    }
    //return x;
     return x_best;
}

vector<double> penalty_method(vector<double>&x,double R,vector<pair<double,double>>bound,int func_choice)
{
    double e1 = 1e-4,C = 10;
//   vector<double> x_c=x;
    vector<double> xprev,xnew;
    xnew = x;
    xprev = x;
    int k=0;
    int M = 100;
    double R_prev;
    while(k<M)
    {
        //  xprev = xnew;
        cout<<k<<" "<<functionvalue(xnew,func_choice)<<endl;
        xnew = Marquardt(xnew,bound,R,e1);
        
      if(k>0){
           if(abs(fun_ptr(xnew,R)-fun_ptr(xprev,R_prev))<=e1){
           return xnew;
           }
      }
       xprev = xnew;
      R_prev=R;
      R = C*R;
       k++; 
    //   cout<<k<<"   ";
    //   cout<< functionvalue(xnew,R)<<"  ";
    //  cout<<endl;
        
    }
    return xnew;
}  

int main()
{
    srand(static_cast<unsigned int>(time(NULL)));
    int n;     // no.of variables
    int func_choice;
    cout << "Select a function ( 1  2  3): ";
    cin >> func_choice;
    double R = 0.1;
    select_function(func_choice);
    
   
    cout << "Enter number of variables: ";
    cin >> n;


  vector<pair<double,double>>bound(n);
    // vector<pair<double,double>>bound = {
    //     {0,10},
    //     {0,10}
        
    // };
    cout<<"Enter bounds"<<endl;
    for(int i=0;i<n;i++)
    {
        cin>>bound[i].first>>bound[i].second;
    }
   vector<double> x(n,0.0);
for(int i=0;i<10;i++){
   
  //  func_eval = 0;
    
  for (int i = 0; i < n; ++i)
    {
        x[i] = bound[i].first + (bound[i].second - bound[i].first) * ((double)rand() / RAND_MAX);
    }
   
    cout << "Initial guess for x: ";
    for (double xi : x)
    {
        cout << xi << " ";
    }
    cout << endl;
    

    vector<double>ans = penalty_method(x,R,bound,func_choice);

    cout << "Optimized variables: ";

    for (double xi : ans)
    {
        cout << xi << " ";
        
    }
    cout<<endl;
        cout << "Optimized objective function value: " << functionvalue(ans,func_choice)<<endl;
        cout << "Total function evaluations: " << func_eval << endl;
        func_eval++;
    cout << endl;
    cout<<endl;
    this_thread::sleep_for(chrono::seconds(1));
}

    // cout << "Optimized objective function value with penalty: " << fun_ptr(x,R) << endl;
    // cout << "Optimized objective function value: " << functionvalue(ans,func_choice)<<endl;
    // func_eval++;
    // cout << "Total function evaluations: " << func_eval << endl;
    
return 0;
    
}