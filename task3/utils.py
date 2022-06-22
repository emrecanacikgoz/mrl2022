
def calculate_max_char(parse_data):
    src, tgt = [], []
    for word in parse_data:
        src.append(len(word[0])) 
        tgt.append(len(word[1])) 
    print(f"Max src char: {max(src)} among {len(src)} data")
    print(f"Max tgt char: {max(tgt)} among {len(tgt)} data")