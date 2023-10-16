using System;
namespace ValdeML
{
    class DatasetMultFeatures
    {
        Random random = new Random();
        public SCALER[] scalers;
        public MMODEL[] dataset;
        public MMODEL[][] batches;
        public void Build(int datasetsize, int batchsize, double multiplier, string scale_method, bool isbinary)
        {
            dataset = new MMODEL[datasetsize];
            for(int i = 0; i< datasetsize; i++)
            {
                int actual = i + 1;
                MMODEL model = new MMODEL();
                model.input = new double[] { actual * 100, -(actual * 10), actual * 200, actual* Math.Pow(10, -2) };
                //model.input = new double[] { actual };

                if (!isbinary)
                    model.target = actual * multiplier;
                else
                    model.target = actual >= (datasetsize / multiplier) ? 1 : 0;

                dataset[i] = model;
            }
            if (scale_method == "minmax")
            {
                MINMAX scaler = new MINMAX();
                dataset = scaler.Get(dataset);
                scalers = scaler.scalers;
            }
            if (scale_method == "mean")
            {
                MEAN scaler = new MEAN();
                dataset = scaler.Get(dataset);
                scalers = scaler.scalers;
            }
            if (scale_method == "zscore")
            {
                ZSCORE scaler = new ZSCORE();
                dataset = scaler.Get(dataset);
                scalers = scaler.scalers;
            }
            if (scale_method == "maxsin")
            {
                MAXSIN scaler = new MAXSIN();
                dataset = scaler.Get(dataset);
                scalers = scaler.scalers;
            }
            if (scale_method == "maxscose")
            {
                MAXCOS scaler = new MAXCOS();
                dataset = scaler.Get(dataset);
                scalers = scaler.scalers;
            }
            dataset = dataset.OrderBy(_ => random.Next()).ToArray();
            batches = Batches.Get(dataset, batchsize);
        }
    }
    class DatasetOneFeature
    {
        Random random = new Random();
        public SCALER scaler;
        public SMODEL[] dataset;
        public SMODEL[][] batches;
        double GetS(double[] inputs, double average)
        {
            double[] s_calcs = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                double tmp_s = Math.Pow(inputs[i] - average, 2) / (inputs.Length - 1);
                s_calcs[i] = tmp_s;
            }
            return Math.Sqrt(s_calcs.Sum());
        }
        public void Build(int datasetsize, int batchsize, double multiplier, string scale_method, bool isbinary)
        {
            int bid = 0;
            Random random = new Random();
            dataset = new SMODEL[datasetsize];
            for (int i = 0; i < datasetsize; i++)
            {
                int actual = i + 1;
                
                SMODEL model = new SMODEL();
                model.input = actual;

                if (!isbinary)
                    model.target = actual * multiplier;
                else
                    model.target = actual >= (datasetsize / multiplier) ? 1 : 0;

                dataset[i] = model;
            }
            double[] inputs = dataset.Select(x => x.input).ToArray();
            scaler = new SCALER();
            scaler.type = scale_method;
            scaler.s = GetS(inputs, scaler.m);
            scaler.m = inputs.Average();
            scaler.max = inputs.Max();
            scaler.min = inputs.Min();
            for(int i= 0; i< inputs.Length; i++)
            {
                if (scale_method == "minmax")
                    dataset[i].input = (inputs[i] - scaler.min) / (scaler.max - scaler.min);
                else if (scale_method == "mean")
                    dataset[i].input = (inputs[i] - scaler.m) / (scaler.max - scaler.min);
                else if (scale_method == "maxsin")
                {
                    double tmp_calc = ((Math.PI * 2) * inputs[i] / scaler.max);
                    dataset[i].input = Math.Sin(tmp_calc);
                }
                else if (scale_method == "maxcos")
                {
                    double tmp_calc = ((Math.PI * 2) * inputs[i] / scaler.max);
                    dataset[i].input = Math.Cos(tmp_calc);
                }
                else if (scale_method == "zscore")
                    dataset[i].input = (inputs[i] - scaler.m) / scaler.s;
            }
            dataset = dataset.OrderBy(_ => random.Next()).ToArray();

            int total_batches = dataset.Length / batchsize;
            batches = new SMODEL[total_batches][];
            for (int i = 0; i < dataset.Length; i += batchsize)
            {
                if (bid.Equals(total_batches))
                    break;
                SMODEL[] batch = dataset.Skip(i).Take(batchsize).ToArray();
                batches[bid] = batch;
                bid++;
            }
        }
    }
}

