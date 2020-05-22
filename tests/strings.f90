! SPDX-License-Identifier: GPL-2.0+

module strings

    use iso_fortran_env, only: output_unit, real128
    
    implicit none
    
    ! Parameters
    integer, parameter :: dp = selected_real_kind(p=15)
    integer, parameter :: qp = selected_real_kind(p=30)
    integer, parameter :: lp = selected_int_kind(8)
    
    
    character(len=10),parameter :: const_str='1234567890'
    
    character(len=10) :: a_str
    character(len=10) :: a_str_set='abcdefghjk'
    character(:), allocatable :: str_alloc
    
    character(len=10), dimension(5) :: b_str10_exp_1d
    character(len=10), dimension(5,5) :: b_str10_exp_2d

    character(len=10), dimension(:), allocatable :: c_str10_alloc_1d
    character(len=10), dimension(:,:), allocatable :: c_str10_alloc_2d

    
    contains
    
    subroutine sub_str_in_explicit(x)
        character(len=10), intent(in) ::x
        write(output_unit,*) trim(x)
    end subroutine sub_str_in_explicit
    
    subroutine sub_str_in_implicit(x)
        character(len=*), intent(in) ::x
        write(output_unit,*) trim(x)
    end subroutine sub_str_in_implicit
    
    subroutine sub_str_multi(x,y,z)
        integer, intent(in) ::x,z
        character(len=*), intent(in) ::y
        write(output_unit,'(I1,1X,A)') x+z,trim(y)
    end subroutine sub_str_multi
    
    subroutine sub_str_alloc(x_alloc)
        character(:), allocatable, intent(out) :: x_alloc
        x_alloc = 'abcdef'
    end subroutine sub_str_alloc
    
    subroutine sub_str_p(zzz)
        character(len=*),pointer, intent(inout) :: zzz
        
        write(output_unit,'(A)') zzz
        
        zzz = 'xyzxyz'
    end subroutine sub_str_p
      
      
    character(len=5) function func_ret_str(x)
        character(len=5) :: x
        
        func_ret_str = x
        func_ret_str(1:1) = 'A'
        
    end function func_ret_str
    
    
    logical function check_str_alloc(ver)
        integer :: ver
        
        check_str_alloc = .false.
        
        if(allocated(str_alloc)) then
            write(*,*) allocated(str_alloc)
            write(*,*) str_alloc
            if(ver==1) then
                if(str_alloc(1:16)=='abcdefghijklmnop')  check_str_alloc = .true.
            else if (ver==2) then 
                if(str_alloc(1:8)=='12346578')  check_str_alloc = .true.
            end if
        end if
    
    end function check_str_alloc
    
    
    
    function func_str_int_len(i) result(s)
        ! Github issue #12
        integer, intent(in) :: i
        character(len=str_int_len(i)) :: s
        write(s, '(i0)') i
    end function func_str_int_len
          
    pure integer function str_int_len(i) result(sz)
        ! Returns the length of the string representation of 'i'
        integer, intent(in) :: i
        integer, parameter :: MAX_STR = 100
        character(MAX_STR) :: s
        ! If 's' is too short (MAX_STR too small), Fortran will abort with:
        ! "Fortran runtime error: End of record"
        write(s, '(i0)') i
        sz = len_trim(s)
    end function str_int_len
          

    subroutine sub_set_b_str10_exp_1d(s)
        character(len=10), intent(in) :: s
        integer :: i

        do i=lbound(b_str10_exp_1d,dim=1),ubound(b_str10_exp_1d,dim=1)
            b_str10_exp_1d(i) = s
        end do

    end subroutine sub_set_b_str10_exp_1d


    logical function sub_check_b_str10_exp_1d(s)
        character(len=10), intent(in) :: s
        integer :: i

        sub_check_b_str10_exp_1d = .true.
        do i=lbound(b_str10_exp_1d,dim=1),ubound(b_str10_exp_1d,dim=1)
            if(b_str10_exp_1d(i) /= s)  sub_check_b_str10_exp_1d = .false.
        end do

    end function sub_check_b_str10_exp_1d


    subroutine sub_set_b_str10_exp_2d(s)
        character(len=10), intent(in) :: s
        integer :: i,j

        do i=lbound(b_str10_exp_2d,dim=1),ubound(b_str10_exp_2d,dim=1)
            do j=lbound(b_str10_exp_2d,dim=2),ubound(b_str10_exp_2d,dim=2)
                b_str10_exp_2d(i,j) = s
            end do
        end do

    end subroutine sub_set_b_str10_exp_2d


    logical function sub_check_b_str10_exp_2d(s)
        character(len=10), intent(in) :: s
        integer :: i, j

        sub_check_b_str10_exp_2d = .true.
        do i=lbound(b_str10_exp_2d,dim=1),ubound(b_str10_exp_2d,dim=1)
            do j=lbound(b_str10_exp_2d,dim=2),ubound(b_str10_exp_2d,dim=2)
                if(b_str10_exp_2d(i,j) /= s)  sub_check_b_str10_exp_2d = .false.
            end do
        end do

    end function sub_check_b_str10_exp_2d


    subroutine clear_strs()

        b_str10_exp_1d = ''
        b_str10_exp_2d = ''


    end subroutine clear_strs


    subroutine sub_print_assumed(s1,s2)
        character(len=10), dimension(:) :: s1
        character(len=10), dimension(:,:) :: s2

        write(*,*) s1(1),s2(1,1)


    end subroutine sub_print_assumed


    subroutine sub_set_c_str10_alloc_1d(s)
        character(len=10), intent(in) :: s
        integer :: i

        do i=lbound(c_str10_alloc_1d,dim=1),ubound(c_str10_alloc_1d,dim=1)
            c_str10_alloc_1d(i) = s
        end do

    end subroutine sub_set_c_str10_alloc_1d

    subroutine sub_str10_explict_N(N,s)
        integer, intent(in) :: N
        character(len=10),dimension(N), intent(inout) :: s
        integer :: i

        do i=lbound(s,dim=1),ubound(s,dim=1)
            s(i) = 'zxcvbnm'
        end do

    end subroutine sub_str10_explict_N

    subroutine sub_strM_explict_N(N,M,s)
        integer, intent(in) :: N,M
        character(len=M),dimension(N), intent(inout) :: s
        integer :: i

        do i=lbound(s,dim=1),ubound(s,dim=1)
            s(i) = 'zxcvbnm'
        end do

    end subroutine sub_strM_explict_N

    subroutine sub_strStar_explict_N(N,s)
        integer, intent(in) :: N
        character(len=*),dimension(N), intent(inout) :: s
        integer :: i

        do i=lbound(s,dim=1),ubound(s,dim=1)
            s(i) = 'zxcvbnm'
        end do

    end subroutine sub_strStar_explict_N


    logical function sub_check_c_str10_alloc_1d(s)
        character(len=10), intent(in) :: s
        integer :: i

        sub_check_c_str10_alloc_1d = .true.
        do i=lbound(c_str10_alloc_1d,dim=1),ubound(c_str10_alloc_1d,dim=1)
            if(c_str10_alloc_1d(i) /= s)  sub_check_c_str10_alloc_1d = .false.
        end do

    end function sub_check_c_str10_alloc_1d


    subroutine sub_set_c_str10_alloc_2d(s)
        character(len=10), intent(in) :: s
        integer :: i,j

        do i=lbound(c_str10_alloc_2d,dim=1),ubound(c_str10_alloc_2d,dim=1)
            do j=lbound(c_str10_alloc_2d,dim=2),ubound(c_str10_alloc_2d,dim=2)
                c_str10_alloc_2d(i,j) = s
            end do
        end do

    end subroutine sub_set_c_str10_alloc_2d


    logical function sub_check_c_str10_alloc_2d(s)
        character(len=10), intent(in) :: s
        integer :: i, j

        sub_check_c_str10_alloc_2d = .true.
        do i=lbound(c_str10_alloc_2d,dim=1),ubound(c_str10_alloc_2d,dim=1)
            do j=lbound(c_str10_alloc_2d,dim=2),ubound(c_str10_alloc_2d,dim=2)
                if(c_str10_alloc_2d(i,j) /= s)  sub_check_c_str10_alloc_2d = .false.
            end do
        end do

    end function sub_check_c_str10_alloc_2d


    subroutine clear_strs_alloc()

        c_str10_alloc_1d = ''
        c_str10_alloc_2d = ''


    end subroutine clear_strs_alloc

    subroutine sub_dealloc_strs

        if(allocated(c_str10_alloc_1d)) deallocate(c_str10_alloc_1d)
        if(allocated(c_str10_alloc_2d)) deallocate(c_str10_alloc_2d)

    end subroutine sub_dealloc_strs

    subroutine sub_alloc_strs

        if(.not.allocated(c_str10_alloc_1d)) allocate(c_str10_alloc_1d(10))
        if(.not.allocated(c_str10_alloc_2d)) allocate(c_str10_alloc_2d(10,10))

    end subroutine sub_alloc_strs


    subroutine sub_print_exp(s1,s2, s3)
        character(len=10), dimension(:) :: s1
        character(len=10), dimension(:,:) :: s2
        character(len=*), dimension(:,:) :: s3

        write(*,*) s1(1),s2(1,1),s3(1,1)


    end subroutine sub_print_exp

end module strings
