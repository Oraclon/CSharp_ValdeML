using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    internal class BCS : iBCSF
    {
        public double[] Prediction(Grad grad, double input)
        {
            throw new NotImplementedException();
        }
        //Experimental
        public double OptimizeW(Grad grad)
        {
            throw new NotImplementedException();
        }
        public double OptimizeB(Grad grad)
        {
            throw new NotImplementedException();
        }
        //Experimental
        public double[] InputDerivatives(Grad grad, double[] inputs)
        {
            throw new NotImplementedException();
        }
        public double[] PredictionsDerivatives(Grad grad)
        {
            throw new NotImplementedException();
        }
        public double[] ErrorDerivatives(Grad grad, double[] targets)
        {
            throw new NotImplementedException();
        }
        public double[] Errors(Grad grad, double[] targets)
        {
            throw new NotImplementedException();
        }
        public double SigmoidActivation(double prediction)
        {
            throw new NotImplementedException();
        }
        public double[] Predictions(Grad grad, double[] inputs)
        {
            throw new NotImplementedException();
        }
        public void Train(Grad grad, SMODEL[][] batches)
        {
            throw new NotImplementedException();
        }
    }
}
