namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            Dataset dataSet = new Dataset();
            dataSet.BuildDemo(20000, 256, 4, Scaler.ZScore, false);

            BinaryClassification ml = new BinaryClassification();
            ml.model.Learning = 0.4;
            ml.Train(dataSet);
        }
    }
}
