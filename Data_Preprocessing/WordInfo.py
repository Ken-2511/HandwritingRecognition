# this is a class which would be used by many files,
# Thus I made it a separate file

class WordInfo:
    def __init__(self, word_id, is_ok, grey, x, y, w, h, gram, word):
        """
        (copied from words.txt)
        format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
        
        a01-000u-00-00  -> word id for line 00 in form a01-000u
        ok              -> result of word segmentation
                            ok: word was correctly
                            er: segmentation of word can be bad
        
        154             -> graylevel to binarize the line containing this word
        408 768 27 51   -> bounding box around this word in x,y,w,h format
        AT              -> the grammatical tag for this word, see the
                            file tagset.txt for an explanation
        A               -> the transcription for this word
        """
        self.word_id = str(word_id)
        self.is_ok = True if is_ok == "ok" else False
        self.grey = int(grey)
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.gram = str(gram)
        self.word = str(word)
    
    def __str__(self) -> str:
        return f"WordInfo({self.word_id}, {self.is_ok}, {self.grey}, {self.x}, {self.y}, {self.w}, {self.h}, {self.gram}, {self.word})"
    
    def __repr__(self) -> str:
        return self.__str__()
