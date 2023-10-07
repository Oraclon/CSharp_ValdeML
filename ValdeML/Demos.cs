using System;
namespace ValdeML
{
    class DatasetMultFeatures
    {
        Transposer transposer = new Transposer();
        Random random = new Random();
        public SCALER[] scalers;
        public MMODEL[] dataset;
        public MMODEL[][] batches;
        public void Build(int datasetsize, int batchsize, double multiplier, string scale_method)
        {
            dataset = new MMODEL[datasetsize];
            for(int i = 0; i< datasetsize; i++)
            {
                int x = i + 1;
                MMODEL model = new MMODEL();
                model.input = new double[] { x * 100, x * 10 };
                model.target = x * multiplier;
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
            batches = new Batches().Get(dataset, batchsize);
        }
    }
}

