using System;
namespace ValdeML
{
    public class Transposer
    {
        public double[][] TransposeList(double[][] inputs)
        {
            double[][] transposed = new double[inputs[0].Length][];
            for(int i= 0; i < inputs[0].Length; i++)
            {
                double[] input_array = new double[inputs.Length];
                for(int j= 0; j< inputs.Length; j++)
                {
                    input_array[j] = inputs[j][i];
                }
                transposed[i] = input_array;
            }
            return transposed;
        }
    }
}

