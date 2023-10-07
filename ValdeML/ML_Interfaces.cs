using System;
namespace ValdeML
{
	public interface IML
	{
		double[] Errors(Grad grad, double[] targets);
		double[] ErrorDerivatives(Grad grad, double[] targets);
		double[] InputDerivatives(Grad grad, double[] inputs);
		double[] OptimizeW(Grad grad);
		double[] OptimizeB(Grad grad);
	}
}

