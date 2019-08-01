from django.db import models

# 장고 튜토리얼을 위한 모델 생성
class Member(models.Model):
    name = models.CharField(max_length=200)
    mail = models.CharField(max_length=200)
    age = models.IntegerField(default=0)
