import stylometry as st
def analyze_string(sentence):
    """
    List of tuples of some of the stylometry analysis output
    """
    style = st.extract.StyloString(sentence)
    out = zip(style.csv_header().split(','), style.csv_output().split(','))
    res = [a for a in out]

    # relevant stuff
    idx =[2, 3, 4, 5, 6, 7]
    vals = [res[i] for i in idx]
    return [ {'metric':metric, 'score': value} for metric,value in vals]

