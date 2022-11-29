import copy
from dataclasses import dataclass
from datasets import ClassLabel, Features, TaskTemplate, Value
from typing import ClassVar, Dict, Union


@dataclass(frozen=True)
class TextTemplate(TaskTemplate):
    """Dataset casting for single sentences tasks.

    Note
    ----
        Since HF provides `TextClassification` task template that
        accepts only `ClassLabel` as label column, this template
        implements this template without this restriction.
    """
    task: str = ""
    text_column: str = "text"
    label_column: str = "label"

    input_schema: ClassVar[Features] = Features({
        "text": Value("string")
    })

    label_schema: ClassVar[Features] = Features({
        "label": Union[ClassLabel, int, float]
    })

    def _sanitize_features(self, features):
        return True

    def align_with_features(self, features):
        self._sanitize_features(features)

        # update label schema to reflect label feature
        label_schema = self.label_schema.copy()
        label_schema["label"] = features[self.label_column]

        # updated task template
        task_template = copy.deepcopy(self)
        task_template.__dict__['label_schema'] = label_schema

        return task_template

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.text_column: "text",
            self.label_column: "label"
        }


@dataclass(frozen=True)
class TextPairTemplate(TaskTemplate):
    """Dataset casting for pair of sentences tasks.

    Note
    ----
        Since HF does not provide a classification template
        for pair of sentences, this class implements it. It
        is based on `TextClassification` task template class
        available at:

        https://github.com/huggingface/datasets/blob/main/src/datasets/tasks/text_classification.py
    """
    task: str = ""
    text_column: str = "text"
    text_pair_column: str = "text_pair"
    label_column: str = "label"

    input_schema: ClassVar[Features] = Features({
        "text": Value("string"),
        "text_pair": Value("string")
    })

    label_schema: ClassVar[Features] = Features({
        "label": ClassLabel
    })

    def _sanitize_features(self, features):
        return True

    def align_with_features(self, features):
        self._sanitize_features(features)

        # update label schema to reflect label feature
        label_schema = self.label_schema.copy()
        label_schema["label"] = features[self.label_column]

        # updated task template
        task_template = copy.deepcopy(self)
        task_template.__dict__['label_schema'] = label_schema

        return task_template

    @property
    def column_mapping(self) -> Dict[str, str]:
        return {
            self.text_column: "text",
            self.text_pair_column: "text_pair",
            self.label_column: "label"
        }
