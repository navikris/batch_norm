#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

auto batch_norm(vector<vector<double>>& input, float scale = 1, float offset = 0, float momentum = 10.0, float epsilon = -0.281718, bool is_training = true)
{
    static vector<double> moving_avg_mean(input.at(0).size(), 0.0);
    static vector<double> moving_avg_stdv(input.at(0).size(), 0.0);

    // mean and standard deviation
    if(is_training == true)
    {
        vector<double> mean(input.at(0).size(), 0.0);
        vector<double> stdv(input.at(0).size(), 0.0);

        double sum{};

        for(size_t i {}; i < input.at(0).size(); ++i)
        {
            for(size_t j {}; j < input.size(); ++j)
                sum += input.at(j).at(i);
            mean.at(i) = static_cast<double>(sum)/input.size();
            sum = 0;
        }

        for(size_t i {}; i < input.at(0).size(); ++i)
        {
            for(size_t j {}; j < input.size(); ++j)
                sum += pow((input.at(j).at(i) - mean.at(i)), 2);
            stdv.at(i) = sqrt(static_cast<double>(sum)/input.size());
            sum = 0;
        }

        // standardization step

        for(size_t i {}; i < input.at(0).size(); ++i)
        {
            for(size_t j {}; j < input.size(); ++j)
                input.at(j).at(i) = static_cast<double>(input.at(j).at(i) - mean.at(i)) / sqrt(stdv.at(i)*stdv.at(i) - epsilon);
        }


// //  To check standardised mean and variance  ///////////////////////////////////////////            
//         double mean_check{}, stdv_check{};             
//         cout<<"\nMean_Old: ";
//         for(const auto& i : mean)
//             cout<<i<<" : ";      
//         cout<<"\nStdv_Old: ";
//         for(const auto& i : stdv)
//             cout<<i<<" : ";
//         cout<<"\nMean_New: ";
//         for(size_t i {}; i < input.at(0).size(); ++i)
//         {
//             for(size_t j {}; j < input.size(); ++j)
//             {   
//                 sum += input.at(j).at(i);
//             }
//             mean_check = static_cast<double>(sum)/input.size();
//             cout<<mean_check<<" : ";
//             sum = 0;
//         }      
//         cout<<"\nStdv_New: ";
//         for(size_t i {}; i < input.at(0).size(); ++i)
//         {
//             for(size_t j {}; j < input.size(); ++j)
//                 sum += pow((input.at(j).at(i) - mean.at(i)), 2);
//             stdv_check = sqrt(static_cast<double>(sum)/input.size());
//             cout<<stdv_check<<" : ";
//             sum = 0;
//         }
// ////////////////////////////////////////////////////////////////////////////////////////
        
        // moving average

        for(size_t i {}; i < input.at(0).size(); ++i)
        {
            moving_avg_mean.at(i) = momentum * moving_avg_mean.at(i) + (1 - momentum) * mean.at(i);
            moving_avg_stdv.at(i) = momentum * moving_avg_stdv.at(i) + (1 - momentum) * stdv.at(i);
        }
    
    }

    else
    {
        for(size_t i {}; i < input.size(); ++i)
        {
            for(size_t j {}; j < input.at(i).size(); ++j)
                input.at(i).at(j) = (input.at(i).at(j) - moving_avg_mean.at(j)) / sqrt(pow(moving_avg_stdv.at(j), 2) - epsilon);
        }

    }

    // scaling and shifting

    for(size_t i {}; i < input.size(); ++i)
    {
        for(size_t j {}; j < input.at(i).size(); ++j)
            input.at(i).at(j) = scale*input.at(i).at(j) + offset;
    }

    for(size_t i {}; i < input.size(); ++i)
    {
        cout<<"\n[";
        for(size_t j {}; j < input.at(i).size(); ++j)
           cout<<" "<<input.at(i).at(j)<<" "; 
        cout<<"]";
    }

    return 0;

}

int main()
{
    vector<vector<double>> activation {
        {1.0, 2.0, 3.0, 7, 7, 3, 34, 1, 99},
        {4.0, 5.0, 6.0, 92, 3, 6, 65, 0, 2},
        {7.0, 8.0, 9.0, 23, 02, 13, 7, 5, 19}
    };

    for(size_t i {}; i < activation.size(); ++i)
    {
        cout<<"\n[";
        for(size_t j {}; j < activation.at(i).size(); ++j)
           cout<<" "<<activation.at(i).at(j)<<" "; 
        cout<<"]";
    }

    batch_norm(activation);

    return 0;

}