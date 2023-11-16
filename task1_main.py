class SampleInfo(object):

    def __init__(self):
        self.sample_name = "sample1"
        self.exposure_time = ""
        self.angle = ""

    def set_sample_name(self, sample_name):
        self.sample_name = sample_name

    def get_sample_name(self):
        return self.sample_name

    def set_exposure_time(self, exposure_time):
        self.exposure_time = exposure_time

    def get_exposure_time(self):
        return self.exposure_time

    def set_angle(self, angle):
        self.angle = angle

    def get_angle(self):
        return self.angle


def parse_ner(ner_objs, sample_info):

    for ner_obj in ner_objs:
        if 'sample_name' in ner_obj:
            sample_info.set_sample_name(ner_obj["sample_name"])
        if 'exposure_time' in ner_obj:
            sample_info.set_exposure_time(ner_obj["exposure_time"])
        if 'angle' in ner_obj:
            sample_info.set_angle(ner_obj["angle"])

    return sample_info


def task1_bot(ex_feature, ner_obj):
    sample_info = SampleInfo()
    sample_info = parse_ner(ner_obj, sample_info)

    while not sample_info.get_exposure_time() or not sample_info.get_angle():
        print("Processing sample info. sample_name: {}, exposure_time: {}, angle {}"
              .format(sample_info.get_sample_name(), sample_info.get_exposure_time(), sample_info.get_angle()))
        query = ""
        if not sample_info.get_exposure_time() and not sample_info.get_angle():
            query = input("What should be the sample exposure time and incident angle?: ")
        elif not sample_info.get_exposure_time():
            query = input("What should be the exposure time for this experiment?: ")
        elif not sample_info.get_angle():
            query = input("What should be the incident angle for this experiment?: ")

        result = ex_feature.predict_ner(query)
        sample_info = parse_ner(result['ner'], sample_info)

        if query == "quit":
            break

    print("Running experiments with sample_name: {}, exposure_time: {}, and incident angle of {}"
          .format(sample_info.get_sample_name(), sample_info.get_exposure_time(), sample_info.get_angle()))





