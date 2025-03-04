public static class Helpers
{
  public static Int16 DateToNumber(this DateTime time)
  {
    DateTime base = new DateTime(1980,1,1,0,0,0, time.Kind);
    int epoch = (time-base).Days;
    return (Int16)epoch;
  }
}
