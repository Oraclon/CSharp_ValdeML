using System;
using ValdeML;

namespace ValdeML
{
	public class LRS : iLRSF
	{
		

		

		

		public double OptimizeB(Grad grad)
		{
			throw new NotImplementedException();
		}

		public double OptimizeW(Grad grad)
		{
			throw new NotImplementedException();
		}

		public double Predict(Grad grad, double input)
		{
			throw new NotImplementedException();
		}

        public double[] InputDerivatives(Grad grad, double[] inputs)
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

