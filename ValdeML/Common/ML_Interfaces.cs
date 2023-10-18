using System;
namespace ValdeML
{
    public interface iEval
    {
        double[] Evaluate(double[] features);
    }
    public interface iML : iEval
    {
        void Train(Dataset data);
        void _Predict(double[][] inputs);
        void _Errors(double[] targets);
        void _CalculateDeltas(double[][] respect_to);
        void _BackPropagate();
        void _GenerateGradient();
    }
}