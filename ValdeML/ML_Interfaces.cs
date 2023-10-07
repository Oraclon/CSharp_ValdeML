using System;
namespace ValdeML
{
    public interface iScaler
    {
        MMODEL[] Get(MMODEL[] dataset);
        SCALER[] GetScalers(double[][] inputs, string type);
        MMODEL[] Calc(MMODEL[] dataset, double[][] inputs);
    }
    //Common Functions for ML
	public interface iML
	{
		double[] Errors(Grad grad, double[] targets);
		double[] ErrorDerivatives(Grad grad, double[] targets);
		double[] InputDerivatives(Grad grad, double[] inputs);
		double OptimizeW(Grad grad);
		double OptimizeB(Grad grad);
	}
	//Linear Regression Single Feature.
	public interface iLRSF: iML
	{
		void Train(Grad grad, SMODEL[][] batches);
		double Predict(Grad grad, double input);
		double[] Predictions(Grad grad, double[] inputs);
	}
    //Linear Regression Multiple Features.
    public interface iLRMF: iML
	{
        void Train(Grad grad, MMODEL[][] batches);
        double Predict(Grad grad, double[] inputs);
		double[] Predictions(Grad grad, double[][] inputs);
    }
	public interface iBC: iML
	{
		double SigmoidActivation(double prediction);
	}
	//Binary Classification Single Feature.
	public interface iBCSF: iBC
	{
		void Train(Grad grad, SMODEL[][] batches);
		double[] Predictions(Grad grad, double[] inputs);
		double Prediction(Grad grad, double input);
	}
    //Binary Classification Multiple Features.
    public interface iBCMG: iBC
	{
		void Train(Grad grad, MMODEL[][] batches);
		double[] Predictions(Grad grad, double[][] inputs);
		double Prediction(Grad grad, double[] inputs);
	}
}

