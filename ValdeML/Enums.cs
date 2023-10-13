using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    public enum Activation
    {
        None,      //0
        ReLU,      //1
        LeakyReLU, //2
        Tanh,      //3
        Sigmoid,   //4
        SoftMax    //5
    }
    public enum Scaler
    {
        MinMax,    //0
        Mean,      //1
        ZScore,    //2
        MaxSin,    //3
        MaxCos     //4
    }
    public enum Errors
    {
        Mean,      //0
        LogLoss,   //1
    }
    public enum Optimizers
    { 
        Momentum,  //0
        RmsProp,   //1
        Adam       //2
    }
}
