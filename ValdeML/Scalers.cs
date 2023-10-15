using System;
using System.Reflection.Metadata.Ecma335;

namespace ValdeML
{
    public class MINMAX : iScaler
    {
        public SCALER[] scalers;
        public MMODEL[] Calc(MMODEL[] dataset, double[][] inputs)
        {
            double[][] scaled_lst = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = scalers[i];
                double[] inps = inputs[i];
                double[] scaled = new double[inps.Length];
                for (int j = 0; j < inps.Length; j++)
                {
                    scaled[j] = (inps[j] - scaler.min) / (scaler.max - scaler.min);
                }
                scaled_lst[i] = scaled;
            }
            double[][] retransposed = Transposer.TransposeList(scaled_lst);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }

        public MMODEL[] Get(MMODEL[] dataset)
        {
            scalers = new SCALER[dataset[0].input.Length];
            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = Transposer.TransposeList(inputs);
            scalers = GetScalers(inputsT, "minmax");
            return Calc(dataset, inputsT);
        }

        public SCALER[] GetScalers(double[][] inputs, string type)
        {
            SCALER[] scalers_lst = new SCALER[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = type;
                scaler.max = inputs[i].Max();
                scaler.min = inputs[i].Min();
                scalers_lst[i] = scaler;
            }
            return scalers_lst;
        }
    }

    public class MEAN : iScaler
    {
        public SCALER[] scalers;
        public MMODEL[] Calc(MMODEL[] dataset, double[][] inputs)
        {
            double[][] scaled_lst = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = scalers[i];
                double[] inps = inputs[i];
                double[] scaled = new double[inps.Length];
                for (int j = 0; j < inps.Length; j++)
                {
                    scaled[j] = (inps[j] - scaler.m) / (scaler.max - scaler.min);
                }
                scaled_lst[i] = scaled;
            }
            double[][] retransposed = Transposer.TransposeList(scaled_lst);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }

        public MMODEL[] Get(MMODEL[] dataset)
        {
            scalers = new SCALER[dataset[0].input.Length];
            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = Transposer.TransposeList(inputs);
            scalers = GetScalers(inputsT, "mean");
            return Calc(dataset, inputsT);
        }

        public SCALER[] GetScalers(double[][] inputs, string type)
        {
            SCALER[] scalers_lst = new SCALER[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = type;
                scaler.m = inputs[i].Average();
                scaler.max = inputs[i].Max();
                scaler.min = inputs[i].Min();
                scalers_lst[i] = scaler;
            }
            return scalers_lst;
        }
    }

    public class ZSCORE : iScaler
    {
        public SCALER[] scalers;
        public MMODEL[] Get(MMODEL[] dataset)
        {
            scalers = new SCALER[dataset[0].input.Length];
            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = Transposer.TransposeList(inputs);
            scalers = GetScalers(inputsT, "zscore");
            return Calc(dataset, inputsT);
        }
        public SCALER[] GetScalers(double[][] inputs, string type)
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
                scaler.type = type;
                scaler.m = inputs[i].Average();
                scaler.s = GetS(inputs[i], scaler.m);
                scalers_lst[i] = scaler;
            }
            return scalers_lst;
        }
        public MMODEL[] Calc(MMODEL[] dataset, double[][] inputs)
        {
            Random r = new Random();
            double[][] scaled_lst = new double[inputs.Length][];
            for(int i= 0; i< inputs.Length; i++)
            {
                double test = r.NextDouble();
                SCALER scaler = scalers[i];
                double[] inps = inputs[i];
                double[] scaled = new double[inps.Length];
                for(int j= 0; j< inps.Length; j++)
                {
                    scaled[j] = !scaler.m.Equals(0) ? (inps[j] - scaler.m) / scaler.s : (scaler.s - scaler.m) / scaler.s;
                }
                scaled_lst[i] = scaled;
            }
            double[][] retransposed = Transposer.TransposeList(scaled_lst);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            dataset.OrderBy(_ => r.Next()).ToArray();
            return dataset;
        }
    }

    public class MAXSIN : iScaler
    {
        public SCALER[] scalers;
        public MMODEL[] Calc(MMODEL[] dataset, double[][] inputs)
        {
            double[][] scaled_lst = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = scalers[i];
                double[] inps = inputs[i];
                double[] scaled = new double[inps.Length];
                for (int j = 0; j < inps.Length; j++)
                {
                    double tmp_calc = ((2 * Math.PI) * inps[j]) / scaler.max;
                    scaled[j] = Math.Sin(tmp_calc);
                }
                scaled_lst[i] = scaled;
            }
            double[][] retransposed = Transposer.TransposeList(scaled_lst);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }

        public MMODEL[] Get(MMODEL[] dataset)
        {
            scalers = new SCALER[dataset[0].input.Length];
            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = Transposer.TransposeList(inputs);
            scalers = GetScalers(inputsT, "maxsin");
            return Calc(dataset, inputsT);
        }

        public SCALER[] GetScalers(double[][] inputs, string type)
        {
            SCALER[] scalers_lst = new SCALER[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = type;
                scaler.max = inputs[i].Max();
                scalers_lst[i] = scaler;
            }
            return scalers_lst;
        }
    }

    public class MAXCOS : iScaler
    {
        public SCALER[] scalers;
        public MMODEL[] Calc(MMODEL[] dataset, double[][] inputs)
        {
            double[][] scaled_lst = new double[inputs.Length][];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = scalers[i];
                double[] inps = inputs[i];
                double[] scaled = new double[inps.Length];
                for (int j = 0; j < inps.Length; j++)
                {
                    double tmp_calc = ((2 * Math.PI) * inps[j]) / scaler.max;
                    scaled[j] = Math.Cos(tmp_calc);
                }
                scaled_lst[i] = scaled;
            }
            double[][] retransposed = Transposer.TransposeList(scaled_lst);
            for (int i = 0; i < retransposed.Length; i++)
            {
                dataset[i].input = retransposed[i];
            }
            return dataset;
        }

        public MMODEL[] Get(MMODEL[] dataset)
        {
            scalers = new SCALER[dataset[0].input.Length];
            double[][] inputs = dataset.Select(x => x.input).ToArray();
            double[][] inputsT = Transposer.TransposeList(inputs);
            scalers = GetScalers(inputsT, "maxcos");
            return Calc(dataset, inputsT);
        }

        public SCALER[] GetScalers(double[][] inputs, string type)
        {
            SCALER[] scalers_lst = new SCALER[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                SCALER scaler = new SCALER();
                scaler.type = type;
                scaler.max = inputs[i].Max();
                scalers_lst[i] = scaler;
            }
            return scalers_lst;
        }
    }
}

