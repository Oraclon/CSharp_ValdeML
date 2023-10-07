using System;
namespace ValdeML
{
	public interface iML
	{
		double[] Errors(Grad grad, double[] targets);
		double[] ErrorDerivatives(Grad grad, double[] targets);
		double[] InputDerivatives(Grad grad, double[] inputs);
		double[] OptimizeW(Grad grad);
		double[] OptimizeB(Grad grad);
	}
	public interface iScaler
	{
		MMODEL[] Get(MMODEL[] dataset);
		SCALER[] GetScalers(double[][] inputs, string type);
		MMODEL[] Calc(MMODEL[] dataset, double[][] inputs);
	}
}

