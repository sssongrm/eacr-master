from django.db import models

# Create your models here.

class Proposal(models.Model):
    # id = models.AutoField(primary_key=True, null=False, max_length=11, unique=True)  # 自增id，设置主键
    text = models.CharField(null=False, max_length=255)
    time = models.CharField(null=False, max_length=255)

    class Meta:
        db_table = 'proposal'  # 表名

class Brand(models.Model):
    # id = models.AutoField(primary_key=True, null=False, max_length=11, unique=True)  # 自增id，设置主键
    brand = models.CharField(null=False, max_length=255)
    type = models.CharField(null=False, max_length=255)

    class Meta:
        db_table = 'brand'  # 表名