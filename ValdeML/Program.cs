namespace ValdeML
{
    class Program
    {
        static void Main(string[] args)
        {
            Dataset dataSet = new Dataset();
            dataSet.BuildDemo(300000, 16, 2, Scaler.ZScore, true);

            BinaryClassification ml = new BinaryClassification(.4, 0);
            ml.Train(dataSet);
        }
    }
}
