using System;
using System.Reflection.Metadata.Ecma335;

namespace ValdeML
{
    //To be rewritten
    public class MINMAX
    {
        public SCALER[] scalers;
        public MMODEL[] Get(MMODEL[] dataset)
        {
            void Calc(SCALER scaler, double[] input)
            {
                for (int x = 0; x < input.Length; x++)
                {
                    input[x] = (input[x] - scaler.min) / (scaler.max - scaler.min);
                }
            }

            Transposer transposer = new Transposer();

            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = transposer.TransposeList(inputs);

            //Collect Scalers
            scalers = new SCALER[inputsT.Length];
            for (int x = 0; x < inputsT.Length; x++)
            {
                SCALER scaler = new SCALER();
                scaler.type = "minmax";
                scaler.min = inputsT[x].Min();
                scaler.max = inputsT[x].Max();
                scalers[x] = scaler;

                //Do calculations based on inputsT[x];
                Calc(scaler, inputsT[x]);
            }
            double[][] retransposed = transposer.TransposeList(inputsT);
            for (int x = 0; x < retransposed.Length; x++)
            {
                dataset[x].input = retransposed[x];
            }
            return dataset;
        }
    }
    public class MEAN
    {
        public SCALER[] scalers;
        public MMODEL[] Get(MMODEL[] dataset)
        {
            void Calc(SCALER scaler, double[] input)
            {
                for (int x = 0; x < input.Length; x++)
                {
                    input[x] = (input[x] - scaler.m) / (scaler.max - scaler.min);
                }
            }

            Transposer transposer = new Transposer();

            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = transposer.TransposeList(inputs);

            //Collect Scalers
            scalers = new SCALER[inputsT.Length];
            for (int x = 0; x < inputsT.Length; x++)
            {
                SCALER scaler = new SCALER();
                scaler.type = "minmax";
                scaler.m = inputs[x].Average();
                scaler.min = inputsT[x].Min();
                scaler.max = inputsT[x].Max();
                scalers[x] = scaler;

                //Do calculations based on inputsT[x];
                Calc(scaler, inputsT[x]);
            }
            double[][] retransposed = transposer.TransposeList(inputsT);
            for (int x = 0; x < retransposed.Length; x++)
            {
                dataset[x].input = retransposed[x];
            }
            return dataset;
        }
    }
    public class ZSCORE
    {
        public SCALER[] scalers;
        public MMODEL[] Get(MMODEL[] dataset)
        {
            void Calc(SCALER scaler, double[] input)
            {
                for (int x = 0; x < input.Length; x++)
                {
                    input[x] = (input[x] - scaler.m) / scaler.s;
                }
            }

            Transposer transposer = new Transposer();

            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = transposer.TransposeList(inputs);

            //Collect Scalers
            scalers = new SCALER[inputsT.Length];
            for (int x = 0; x < inputsT.Length; x++)
            {
                SCALER scaler = new SCALER();
                scaler.type = "minmax";
                scaler.m = inputs[x].Average();
                double[] s_tmps = new double[inputsT[x].Length];
                double[] inps = inputsT[x];
                for(int z= 0; z< inps.Length; z++)
                {
                    s_tmps[z] = Math.Pow(inps[z] - scaler.m, 2) / (inps.Length - 1);
                }
                scaler.s = Math.Sqrt(s_tmps.Sum());
                scalers[x] = scaler;

                //Do calculations based on inputsT[x];
                Calc(scaler, inputsT[x]);
            }
            double[][] retransposed = transposer.TransposeList(inputsT);
            for (int x = 0; x < retransposed.Length; x++)
            {
                dataset[x].input = retransposed[x];
            }
            return dataset;
        }
    }
}

