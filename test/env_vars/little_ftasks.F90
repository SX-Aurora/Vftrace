MODULE little_ftasks
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: collatz_max_steps, &
             pythagorean_triples, &
             largest_prime
CONTAINS
   INTEGER FUNCTION collatz_max_steps(nmax) RESULT(max_steps)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: nmax
      INTEGER :: n, i, steps
      max_steps = 0
      DO n = 1, nmax
         i = n
         steps = 0
         DO WHILE (i /= 1)
            IF (MOD(i,2) == 0) THEN
               i = i/2
            ELSE
               i = 3*i+1
            END IF
            steps = steps+1
         END DO
         IF (steps > max_steps) THEN
            max_steps = steps
         END IF
      END DO
      RETURN     
   END FUNCTION collatz_max_steps

   INTEGER FUNCTION pythagorean_triples(n) RESULT(largest)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: n
      INTEGER :: a, b, c, csq
      largest = 0
      DO a = 1, n-1
         DO b = a, n-1
            csq = a*a + b*b
            c = INT(SQRT(1.0*csq))
            IF (c*c == csq .AND. csq > largest) THEN
               largest = csq
            END IF
         END DO
      END DO
      RETURN
   END FUNCTION pythagorean_triples

   INTEGER FUNCTION largest_prime(n) RESULT(largest)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: n
      INTEGER :: i, t, tmax
      LOGICAL :: is_prime
      IF (n<2) THEN
         largest = -1
         RETURN
      END IF
      IF (n==2) THEN
         largest = 2
         RETURN
      END IF
      largest = 0
      DO i = 3, n, 2
         is_prime = .TRUE.
         tmax = INT(SQRT(1.0*i))
         DO t = 3, tmax, 2
            IF (MOD(i,t) == 0) THEN
               is_prime = .FALSE.
               EXIT
            END IF
         END DO
         IF (is_prime) largest = i
      END DO
      RETURN
   END FUNCTION largest_prime
END MODULE little_ftasks

PROGRAM test
   USE, INTRINSIC :: ISO_FORTRAN_ENV
   USE little_ftasks
#ifdef _MPI
   USE mpi
#endif
   IMPLICIT NONE
   INTEGER :: collatz, pythagoras, prime
#ifdef _MPI
   INTEGER :: myrank
   INTEGER :: ierr
   MPI_Init(ierr)
#endif

   collatz = collatz_max_steps(100)
   pythagoras = pythagorean_triples(100)
   prime = largest_prime(100)

#ifdef _MPI
   MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
   IF (myrank == 0) THEN
#endif
      WRITE(OUTPUT_UNIT, '(A,I3)') "collatz: ", collatz
      WRITE(OUTPUT_UNIT, '(A,I5)') "pythagoras: ", pythagoras
      WRITE(OUTPUT_UNIT, '(A,I2)') "prime: ", prime
#ifdef _MPI
   END IF
#endif

#ifdef _MPI
   MPI_Finalize(ierr)
#endif
END PROGRAM test
