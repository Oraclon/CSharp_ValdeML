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
    public class ZSCORE : iScaler
    {
        public SCALER[] scalers;
        Transposer transposer = new Transposer();
        public MMODEL[] Get(MMODEL[] dataset)
        {
            scalers = new SCALER[dataset[0].input.Length];
            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = transposer.TransposeList(inputs);
            scalers = GetScalers(inputsT);
            return Calc(dataset, inputsT);
        }
        public SCALER[] GetScalers(double[][] inputs)
        {
            double GetS(double[] inputs, double average)
            {
                int inputs_size = inputs.Length;
                double[] s_calcs = new double[inputs_size];
                for(int i = 0; i< inputs.Length; i++)
                {
                    double tmp_calc = Math.Pow(inputs[i] - average, 2) / (inputs_size - 1);
                    s_calcs[i] = tmp_calc;
                }
                double s = Math.Sqrt(s_calcs.Sum());
                return s;
            }
            SCALER[] scalers_lst = new SCALER[inputs.Length];
            for(int i = 0; i< inputs.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = "zscore";
                scaler.m = inputs[i].Average();
                scaler.s = GetS(inputs[i], scaler.m);
                scalers_lst[i] = scaler;
            }
            return scalers_lst;
        }
        public MMODEL[] Calc(MMODEL[] dataset, double[][] inputs)
        {
            double[][] scaled_lst = new double[inputs.Length][];
            for(int i= 0; i< inputs.Length; i++)
            {
                SCALER scaler = scalers[i];
                double[] inps = inputs[i];
                double[] scaled = new double[inps.Length];
                for(int j= 0; j< inps.Length; j++)
                {
                    scaled[j] = (inps[j] - scaler.m) / scaler.s;
                }
                scaled_lst[i] = scaled;
            }
            double[][] retransposed = transposer.TransposeList(scaled_lst);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }
    }
    public class MAXSIN
    {
        public SCALER[] scalers;
        public MMODEL[] Get(MMODEL[] dataset)
        {
            void Calc(SCALER scaler, double[] input)
            {
                for (int x = 0; x < input.Length; x++)
                {
                    double tmp_inp = Math.Sin(((2 * Math.PI) * input[x]) / scaler.max) ;
                    input[x] = tmp_inp;
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
                scaler.type = "maxsin";
                scaler.max = inputs[x].Max();
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

