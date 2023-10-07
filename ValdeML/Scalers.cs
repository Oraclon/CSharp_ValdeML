using System;
namespace ValdeML
{
    //To be rewritten
    public class MinMax
    {
        public SCALER[] scalers;
        Transposer transposer = new Transposer();

        public MMODEL[] Get(MMODEL[] dataset)
        {
            void Calc(SCALER scaler, double[] input)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    input[i] = (input[i] - scaler.min) / (scaler.max - scaler.min);
                }
            }
            double[][] inputs = new double[dataset.Length][];
            for (int i = 0; i < dataset.Length; i++)
            {
                inputs[i] = dataset[i].input;
            }

            double[][] inputsT = transposer.TransposeList(inputs);
            scalers = new SCALER[inputsT.Length];
            for (int i = 0; i < inputsT.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = "minmax";
                scaler.min = inputsT[i].Min();
                scaler.max = inputsT[i].Max();
                scalers[i] = scaler;
                Calc(scaler, inputsT[i]);
            }
            double[][] retransposed = transposer.TransposeList(inputsT);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }
    }
    public class Mean
    {
        public SCALER[] scalers;
        Transposer transposer = new Transposer();

        public MMODEL[] Get(MMODEL[] dataset)
        {
            void Calc(SCALER scaler, double[] input)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    input[i] = (input[i] - scaler.m) / (scaler.max - scaler.min);
                }
            }
            double[][] inputs = new double[dataset.Length][];
            for (int i = 0; i < dataset.Length; i++)
            {
                inputs[i] = dataset[i].input;
            }

            double[][] inputsT = transposer.TransposeList(inputs);
            scalers = new SCALER[inputsT.Length];
            for (int i = 0; i < inputsT.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = "mean";
                scaler.min = inputsT[i].Min();
                scaler.max = inputsT[i].Max();
                scaler.m = inputs[i].Average();
                scalers[i] = scaler;
                Calc(scaler, inputsT[i]);
            }
            double[][] retransposed = transposer.TransposeList(inputsT);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }
    }
    public class ZScore
    {
        public SCALER[] scalers;
        Transposer transposer = new Transposer();

        public MMODEL[] Get(MMODEL[] dataset)
        {
            void Calc(SCALER scaler, double[] input)
            {
                for (int i = 0; i < input.Length; i++)
                {
                    input[i] = (input[i] - scaler.m) / scaler.s;
                }
            }
            double[][] inputs = new double[dataset.Length][];
            for (int i = 0; i < dataset.Length; i++)
            {
                inputs[i] = dataset[i].input;
            }

            double[][] inputsT = transposer.TransposeList(inputs);
            scalers = new SCALER[inputsT.Length];
            for (int i = 0; i < inputsT.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = "zscore";
                scaler.m = inputs[i].Average();
                
                Calc(scaler, inputsT[i]);
            }
            double[][] retransposed = transposer.TransposeList(inputsT);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }
    }
}

