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
		double[] Errors(Model grad, double[] targets);
		double[] ErrorDerivatives(Model grad, double[] targets);
		double[] InputDerivatives(Model grad, double[] inputs);
		double OptimizeW(Model grad);
		double OptimizeB(Model grad);
	}
	//Linear Regression Single Feature.
	public interface iLRSF: iML
	{
		void Train(Model grad, SMODEL[][] batches, bool optim_activated);
		double Predict(Model grad, double input);
		double[] Predictions(Model grad, double[] inputs);
	}
    //Linear Regression Multiple Features.
    public interface iLRMF: iML
	{
        void Train(Model grad, MMODEL[][] batches, bool optim_activated);
        double Predict(Model grad, double[] inputs);
		double[] Predictions(Model grad, double[][] inputs);
    }
	public interface iBC: iML
	{
		double SigmoidActivation(double prediction);
	}
	//Binary Classification Single Feature.
	public interface iBCSF: iBC
	{
		void Train(Model grad, SMODEL[][] batches, bool optim_activated);
		double[] Predictions(Model grad, double[] inputs);
		double Predict(Model grad, double input);
	}
    //Binary Classification Multiple Features.
    public interface iBCMF: iBC
	{
		void Train(Model grad, MMODEL[][] batches, bool optim_activated);
		double[] Predictions(Model grad, double[][] inputs);
		double Predict(Model grad, double[] inputs);
	}
}

