"""new foreign keys

Revision ID: 6d28c3f63bb8
Revises: 6748113c1867
Create Date: 2017-09-02 03:42:49.990274

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '6d28c3f63bb8'
down_revision = '6748113c1867'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('lecturers_teaching', sa.Column('programme', sa.String(length=8), nullable=False))
    op.create_foreign_key(None, 'lecturers_teaching', 'programmes', ['programme'], ['program_id'], onupdate='CASCADE',
                          ondelete='CASCADE')
    op.add_column('student_courses', sa.Column('programme', sa.String(length=8), nullable=False))
    op.create_foreign_key(None, 'student_courses', 'programmes', ['programme'], ['program_id'], onupdate='CASCADE',
                          ondelete='CASCADE')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'student_courses', type_='foreignkey')
    op.drop_column('student_courses', 'programme')
    op.drop_constraint(None, 'lecturers_teaching', type_='foreignkey')
    op.drop_column('lecturers_teaching', 'programme')
    # ### end Alembic commands ###
