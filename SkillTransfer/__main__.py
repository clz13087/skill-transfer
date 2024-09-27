from ctypes import windll
from platform import processor

from Processor.Processor import ProcessorClass


def main():
    processor = ProcessorClass()
    processor.mainloop()

if __name__ == "__main__":
    main()
    print("\n----- End program -----")