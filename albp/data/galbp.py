from albp.data.problem import Problem


class GALBP(Problem):

    def __init__(self, params: dict, problem_folder: str, model_folder: str
                 ) -> None:
        super().__init__(params, problem_folder, model_folder)

    def _retrieve_data(self):
        
        pass

    def _write_model(self):
        return super()._write_model()


if __name__ == "__main__":
    pass
