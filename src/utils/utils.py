from string import Formatter

def format_string(info_source, string):
    mapping = {name: getattr(info_source, name) 
               for name in get_format_names(string)}
    
    return  string.format(**mapping)


def get_format_names(string):
    names = [fn for _, fn, _, _ in Formatter().parse(string) 
             if fn is not None]
    return names 
