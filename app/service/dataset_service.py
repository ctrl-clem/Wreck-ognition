from app.repository.dataset_repository import DatasetRepository

class DatasetService():
    def __init__(self, repository:DatasetRepository):
        self._repository = repository

    def get_dataset_json(self):
        return self._repository.get_json_data()

    def get_disaster_types(self):
        return self._repository.get_disaster_types()

    def get_disasters_by_type(self):
        return self._repository.get_disasters_by_type()

    def get_disaster_info(self,disaster_name):
        return self._repository.get_disaster_info(disaster_name)

    def get_disaster_pictures(self,disaster_name):
        return self._repository.get_disaster_pictures(disaster_name)